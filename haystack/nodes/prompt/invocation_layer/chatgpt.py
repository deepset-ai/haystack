import logging
from typing import Any, Dict, List, Optional, Union

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.open_ai import OpenAIInvocationLayer
from haystack.nodes.prompt.invocation_layer.utils import has_azure_parameters
from haystack.utils.openai_utils import (
    _check_openai_finish_reason,
    check_openai_async_policy_violation,
    check_openai_policy_violation,
    count_openai_tokens_messages,
    openai_async_request,
    openai_request,
)

logger = logging.getLogger(__name__)


class ChatGPTInvocationLayer(OpenAIInvocationLayer):
    """
    ChatGPT Invocation Layer

    PromptModelInvocationLayer implementation for OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(
        self,
        api_key: str,
        model_name_or_path: str = "gpt-3.5-turbo",
        max_length: Optional[int] = 500,
        api_base: str = "https://api.openai.com/v1",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
         Creates an instance of ChatGPTInvocationLayer for OpenAI's GPT-3.5 GPT-4 models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param api_key: The OpenAI API key.
        :param api_base: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param kwargs: Additional keyword arguments passed to the underlying model.
        [See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat).
        Note: additional model argument moderate_content will filter input and generated answers for potentially
        sensitive content using the [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
        if set. If the input or answers are flagged, an empty list is returned in place of the answers.
        """
        super().__init__(api_key, model_name_or_path, max_length, api_base=api_base, timeout=timeout, **kwargs)

    def _extract_token(self, event_data: Dict[str, Any]):
        delta = event_data["choices"][0]["delta"]
        if "content" in delta:
            return delta["content"]
        return None

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Make sure the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt

        n_message_tokens = count_openai_tokens_messages(messages, self._tokenizer)
        n_answer_tokens = self.max_length
        if (n_message_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        if isinstance(prompt, str):
            tokenized_prompt = self._tokenizer.encode(prompt)
            n_other_tokens = n_message_tokens - len(tokenized_prompt)
            truncated_prompt_length = self.max_tokens_limit - n_answer_tokens - n_other_tokens

            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fit within the max token limit (%s tokens). "
                "Reduce the length of the prompt to prevent it from being cut off.",
                len(tokenized_prompt),
                truncated_prompt_length,
                n_answer_tokens,
                self.max_tokens_limit,
            )

            truncated_prompt = self._tokenizer.decode(tokenized_prompt[:truncated_prompt_length])
            return truncated_prompt
        else:
            # TODO: support truncation when there is a chat history
            raise ValueError(
                f"The prompt or the messages are too long ({n_message_tokens} tokens). "
                f"The length of the prompt or messages and the answer ({n_answer_tokens} tokens) should be within the max "
                f"token limit ({self.max_tokens_limit} tokens). "
                f"Reduce the length of the prompt or messages."
            )

    @property
    def url(self) -> str:
        return f"{self.api_base}/chat/completions"

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        valid_model = (
            any(m for m in ["gpt-3.5-turbo", "gpt-4"] if m in model_name_or_path)
            and not "gpt-3.5-turbo-instruct" in model_name_or_path
        )
        return valid_model and not has_azure_parameters(**kwargs)

    async def ainvoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. Based on the model, it takes in a prompt (or either a prompt or a list of messages)
        and returns a list of responses using a REST invocation.

        :return: The responses are being returned.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        prompt, base_payload, kwargs_with_defaults, stream, moderation = self._prepare_invoke(*args, **kwargs)

        if moderation and await check_openai_async_policy_violation(input=prompt, headers=self.headers):
            logger.info("Prompt '%s' will not be sent to OpenAI due to potential policy violation.", prompt)
            return []

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt
        else:
            raise ValueError(
                f"The prompt format is different than what the model expects. "
                f"The model {self.model_name_or_path} requires either a string or messages in the ChatML format. "
                f"For more details, see this [GitHub discussion](https://github.com/openai/openai-python/blob/main/chatml.md)."
            )
        extra_payload = {"messages": messages}
        payload = {**base_payload, **extra_payload}
        if not stream:
            response = await openai_async_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=response, payload=payload)
            assistant_response = [choice["message"]["content"].strip() for choice in response["choices"]]
        else:
            response = await openai_async_request(
                url=self.url, headers=self.headers, payload=payload, read_response=False, stream=True
            )
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            assistant_response = self._process_streaming_response(response=response, stream_handler=handler)

        # Although ChatGPT generates text until stop words are encountered, unfortunately it includes the stop word
        # We want to exclude it to be consistent with other invocation layers
        if "stop" in kwargs_with_defaults and kwargs_with_defaults["stop"] is not None:
            stop_words = kwargs_with_defaults["stop"]
            for idx, _ in enumerate(assistant_response):
                for stop_word in stop_words:
                    assistant_response[idx] = assistant_response[idx].replace(stop_word, "").strip()

        if moderation and await check_openai_async_policy_violation(input=assistant_response, headers=self.headers):
            logger.info("Response '%s' will not be returned due to potential policy violation.", assistant_response)
            return []

        return assistant_response

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. Based on the model, it takes in a prompt (or either a prompt or a list of messages)
        and returns a list of responses using a REST invocation.

        :return: The responses are being returned.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        prompt, base_payload, kwargs_with_defaults, stream, moderation = self._prepare_invoke(*args, **kwargs)

        if moderation and check_openai_policy_violation(input=prompt, headers=self.headers):
            logger.info("Prompt '%s' will not be sent to OpenAI due to potential policy violation.", prompt)
            return []

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt
        else:
            raise ValueError(
                f"The prompt format is different than what the model expects. "
                f"The model {self.model_name_or_path} requires either a string or messages in the ChatML format. "
                f"For more details, see this [GitHub discussion](https://github.com/openai/openai-python/blob/main/chatml.md)."
            )
        extra_payload = {"messages": messages}
        payload = {**base_payload, **extra_payload}
        if not stream:
            response = openai_request(url=self.url, headers=self.headers, payload=payload, timeout=self.timeout)
            _check_openai_finish_reason(result=response, payload=payload)
            assistant_response = [choice["message"]["content"].strip() for choice in response["choices"]]
        else:
            response = openai_request(
                url=self.url,
                headers=self.headers,
                payload=payload,
                timeout=self.timeout,
                read_response=False,
                stream=True,
            )
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            assistant_response = self._process_streaming_response(response=response, stream_handler=handler)

        # Although ChatGPT generates text until stop words are encountered, unfortunately it includes the stop word
        # We want to exclude it to be consistent with other invocation layers
        if "stop" in kwargs_with_defaults and kwargs_with_defaults["stop"] is not None:
            stop_words = kwargs_with_defaults["stop"]
            for idx, _ in enumerate(assistant_response):
                for stop_word in stop_words:
                    assistant_response[idx] = assistant_response[idx].replace(stop_word, "").strip()

        if moderation and check_openai_policy_violation(input=assistant_response, headers=self.headers):
            logger.info("Response '%s' will not be returned due to potential policy violation.", assistant_response)
            return []

        return assistant_response
