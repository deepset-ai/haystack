import logging
from typing import Optional, List, Dict, Union, Any

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.open_ai import OpenAIInvocationLayer
from haystack.utils.openai_utils import openai_request, _check_openai_finish_reason, count_openai_tokens_messages

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
        self, api_key: str, model_name_or_path: str = "gpt-3.5-turbo", max_length: Optional[int] = 500, **kwargs
    ):
        super().__init__(api_key, model_name_or_path, max_length, **kwargs)

    def invoke(self, *args, **kwargs):
        """
        It takes in either a prompt or a list of messages and returns a list of responses, using a REST invocation.

        :return: A list of generated responses.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see [OpenAI ChatGPT API reference](https://platform.openai.com/docs/api-reference/chat).
        """
        prompt = kwargs.get("prompt", None)

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

        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            # we use keyword stop_words but OpenAI uses stop
            if "stop_words" in kwargs:
                kwargs["stop"] = kwargs.pop("stop_words")
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
                kwargs["n"] = top_k
                kwargs["best_of"] = top_k
            kwargs_with_defaults.update(kwargs)

        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )
        payload = {
            "model": self.model_name_or_path,
            "messages": messages,
            "max_tokens": kwargs_with_defaults.get("max_tokens", self.max_length),
            "temperature": kwargs_with_defaults.get("temperature", 0.7),
            "top_p": kwargs_with_defaults.get("top_p", 1),
            "n": kwargs_with_defaults.get("n", 1),
            "stream": stream,
            "stop": kwargs_with_defaults.get("stop", None),
            "presence_penalty": kwargs_with_defaults.get("presence_penalty", 0),
            "frequency_penalty": kwargs_with_defaults.get("frequency_penalty", 0),
            "logit_bias": kwargs_with_defaults.get("logit_bias", {}),
        }
        if not stream:
            response = openai_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=response, payload=payload)
            assistant_response = [choice["message"]["content"].strip() for choice in response["choices"]]
        else:
            response = openai_request(
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

        return assistant_response

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

        n_prompt_tokens = count_openai_tokens_messages(messages, self._tokenizer)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        # TODO: support truncation as in _ensure_token_limit methods for other invocation layers
        raise ValueError(
            f"The prompt or the messages are too long ({n_prompt_tokens} tokens). "
            f"The length of the prompt or messages and the answer ({n_answer_tokens} tokens) should be within the max token limit ({self.max_tokens_limit} tokens). "
            f"Reduce the length of the prompt or messages."
        )

    @property
    def url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return model_name_or_path in ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
