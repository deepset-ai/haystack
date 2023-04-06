from typing import Dict, Optional, Any, Tuple, Union, List

import os
import json
import logging

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.preview.nodes.prompt.providers.base import prompt_model_provider
from haystack.preview.nodes.prompt.providers.gpt3 import GPT3Provider


logger = logging.getLogger(__name__)


@prompt_model_provider
class GPT4Provider(GPT3Provider):
    """
    OUsed for OpenAI's GPT-3.5 and GPT-4 models. Invocations are made using REST API.
    See [OpenAI GPT-4](https://platform.openai.com/docs/models/gpt-4) for more details.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_length: Optional[int] = 500,
        azure_base_url: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: str = "2022-12-01",
        default_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3.5 and GPT-4 models.

        :param model_name_or_path: The name or path of the underlying model.
        :param api_key: The OpenAI API key. If empty, Haystack also check if an environment variable called
            `OPENAI_API_KEY` is set and read it from there.
        :param max_length: The maximum length of the output text.
        :param azure_base_url: [Optional for Azure OpenAI] the URL of the Azure OpenAI endpoint.
        :param azure_deployment: [Optional for Azure OpenAI] the name of the deployment.
        :param api_version: [Optional for Azure OpenAI] the API version to use.
        :param default_model_params: Additional parameters to pass to the underlying model by default.
            Relevant parameters include:
                - `suffix`
                - `temperature`
                - `top_p`
                - `presence_penalty`
                - `frequency_penalty`
                - `best_of`
                - `n`
                - `max_tokens`
                - `logit_bias`
                - `stop`
                - `echo`
                - `logprobs`
            Note that `stream` will always be False as this class does not support streaming yet.
            For more details about these parameters, see OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            max_length=max_length,
            azure_base_url=azure_base_url,
            azure_deployment=azure_deployment,
            api_version=api_version,
            default_model_params=default_model_params,
        )

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Returns True if the given model name (with the given arguments) is supported by this provider.

        :param model_name_or_path: the model identifier.
        :param **kwargs: any other argument needed to load this model.
        :returns: True if the model is compatible with this provider, False otherwise.
        """
        return any(m for m in ["gpt-3.5", "gpt-4"] if m in model_name_or_path)

    def _build_payload(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the payload to send to OpenAI.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt
        else:
            raise ValueError(
                f"The prompt format is different than what the model expects. "
                f"The model {self.model_name_or_path} requires either a string or messages in the ChatML format. "
                f"For more details, see this [discussion](https://github.com/openai/openai-python/blob/main/chatml.md)."
            )
        params.pop("best_of", None)
        params.pop("echo", None)
        return {"model": self.model_name_or_path, "messages": messages, **params}

    def _parse_output(self, response: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
        """
        Parses the reply obtained from OpenAI.
        """
        self._check_truncated_answers(result=response)
        assistant_response = [choice["message"]["content"].strip() for choice in response["choices"]]

        # Although ChatGPT generates text until stop words are encountered, unfortunately it includes the stop word
        # We want to exclude it to be consistent with other invocation layers
        if "stop" in params and params["stop"] is not None:
            for idx, _ in enumerate(assistant_response):
                for stop_word in params["stop"]:
                    assistant_response[idx] = assistant_response[idx].replace(stop_word, "").strip()
        return assistant_response

    def ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Make sure the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.
        :param prompt: Prompt text to be sent to the generative model.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt

        n_prompt_tokens = _count_tokens(messages, tokenizer=self.tokenizer)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        # TODO: support truncation as in the ensure_token_limit() methods for other invocation layers
        raise ValueError(
            f"The prompt or the messages are too long ({n_prompt_tokens} tokens). The length of the prompt or messages "
            f"and the answer ({n_answer_tokens} tokens) should be within the max token limit ({self.max_tokens_limit} "
            "tokens). Reduce the length of the prompt or messages."
        )

    def _compile_url_and_headers(
        self, azure_base_url: Optional[str], azure_deployment: Optional[str], api_version: Optional[str]
    ) -> Tuple[str, str]:
        """
        Compiles the URL and headers to use for the API calls
        """
        url, headers = super()._compile_url_and_headers(
            azure_base_url=azure_base_url, azure_deployment=azure_deployment, api_version=api_version
        )
        url = url.replace("/completions", "/chat/completions")
        return url, headers

    def _check_truncated_answers(self, result: Dict[str, Any]) -> None:
        """
        Check the `finish_reason` the answers returned by OpenAI completions endpoint.
        If the `finish_reason` is `length` or `content_filter`, log a warning to the user.

        :param result: The result returned from OpenAI.
        :param payload: The payload sent to OpenAI.
        """
        super()._check_truncated_answers(result=result)

        content_filtered_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "content_filter")
        if content_filtered_completions > 0:
            logger.warning(
                "%s completions have omitted content due to a flag from OpenAI content filters.",
                content_filtered_completions,
            )


def _count_tokens(messages: List[Dict[str, str]], tokenizer: Any) -> int:
    """
    Count the number of tokens in `messages` based on the OpenAI `tokenizer` provided.

    :param messages: The messages to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    :returns: the number of tokens.
    """
    # Adapted from https://platform.openai.com/docs/guides/chat/introduction. Should be kept up to date
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens
