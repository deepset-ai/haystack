from typing import Dict, Optional, Any, Tuple, Union, List

import os
import json
import logging

import requests
from tenacity import retry, wait_exponential, retry_if_not_result
from generalimport import is_imported

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.preview.nodes.prompt.providers.base import prompt_model_provider
from haystack.preview.nodes.prompt.providers.gpt3 import GPT3Provider


logger = logging.getLogger(__name__)


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_OPENAI_TIMEOUT_SEC", 30))


DEFAULT_MODEL_PARAMS = {
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "echo": False,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "best_of": 1,
    "logit_bias": {},
}


@prompt_model_provider
class GPT4Provider:
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
        self.api_key = self._check_api_key(api_key)
        self.model_name_or_path = model_name_or_path
        self.tokenizer, self.max_tokens_limit = self._configure_tokenizer(model=model_name_or_path)
        self.url, self.headers = self._compile_url_and_headers(
            azure_base_url=azure_base_url, azure_deployment=azure_deployment, api_version=api_version
        )
        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16
        self.default_model_params = {**DEFAULT_MODEL_PARAMS, **(default_model_params or {})}
        self.default_model_params["max_tokens"] = self.max_length
        self.default_model_params["stream"] = False  # No support for streaming

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Returns True if the given model name (with the given arguments) is supported by this provider.

        :param model_name_or_path: the model identifier.
        :param **kwargs: any other argument needed to load this model.
        :returns: True if the model is compatible with this provider, False otherwise.
        """
        if not is_imported("requests"):
            logger.debug("'requests' could not be imported. OpenAI GPT-3.5 and GPT-4 models can't be invoked.")
            return False
        return any(m for m in ["gpt-3.5", "gpt-4"] if m in model_name_or_path)

    @retry(retry=retry_if_not_result(bool), wait=wait_exponential(min=1, max=10))
    def invoke(self, prompt: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Sends a prompt the model.

        :param prompt: the prompt to send to the model
        :param model_params: any other parameter needed to invoke this model.
        :return: The responses from the model.
        """
        prompt = self.ensure_token_limit(prompt=prompt)

        all_params = self.default_model_params
        if model_params:
            all_params.update(self._translate_model_parameters(model_params))
        payload = self._build_payload(prompt=prompt, params=all_params)

        response = requests.post(url=self.url, headers=self.headers, data=json.dumps(payload), timeout=OPENAI_TIMEOUT)
        if response.status_code != 200:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        return self._parse_output(result=response.json(), params=all_params)

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
