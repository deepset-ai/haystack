import os
from typing import Dict, List, Union, Optional
import json
import logging

import requests
import requests_cache
import sseclient
from tokenizers import Tokenizer, Encoding

from haystack.errors import AnthropicError, AnthropicRateLimitError, AnthropicUnauthorizedError
from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.utils.requests import request_with_retry
from haystack.environment import HAYSTACK_REMOTE_API_MAX_RETRIES, HAYSTACK_REMOTE_API_TIMEOUT_SEC

ANTHROPIC_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
ANTHROPIC_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))

logger = logging.getLogger(__name__)

# Taken from:
# https://github.com/anthropics/anthropic-sdk-python/blob/main/anthropic/tokenizer.py#L7
# This is a JSON config to load the tokenizer used for Anthropic Claude.
CLAUDE_TOKENIZER_REMOTE_FILE = "https://public-json-tokenization-0d8763e8-0d7e-441b-a1e2-1c73b8e79dc3.storage.googleapis.com/claude-v1-tokenization.json"


class AnthropicClaudeInvocationLayer(PromptModelInvocationLayer):
    """
    Anthropic Claude Invocation Layer
    This layer invokes the Claude API provided by Anthropic.
    """

    def __init__(self, api_key: str, model_name_or_path: str = "claude-v1", max_length=200, **kwargs):
        """
         Creates an instance of PromptModelInvocation Layer for Claude models by Anthropic.
        :param model_name_or_path: The name or path of the underlying model.
        :param max_tokens_to_sample: The maximum length of the output text.
        :param api_key: The Anthropic API key.
        :param kwargs: Additional keyword arguments passed to the underlying model. The list of Anthropic-relevant
        kwargs includes: stop_sequences, temperature, top_p, top_k, and stream. For more details about these kwargs,
        see Anthropic's [documentation](https://console.anthropic.com/docs/api/reference).
        """
        super().__init__(model_name_or_path)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise AnthropicError(
                f"api_key {api_key} must be a valid Anthropic key. Visit https://console.anthropic.com/account/keys to get one."
            )
        self.api_key = api_key
        self.max_length = max_length

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to Anthropic documentation
        supported_kwargs = ["temperature", "top_p", "top_k", "stop_sequences", "stream", "stream_handler"]
        self.model_input_kwargs = {k: v for (k, v) in kwargs.items() if k in supported_kwargs}

        # Number of max tokens is based on the official Anthropic model pricing from:
        # https://cdn2.assets-servd.host/anthropic-website/production/images/FINAL-PRICING.pdf
        self.max_tokens_limit = 9000
        self.tokenizer: Tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> Tokenizer:
        # Expire cache after a day
        expire_after = 60 * 60 * 60 * 24
        # Cache the JSON config to avoid downloading it each time as it's a big file
        with requests_cache.enabled(expire_after=expire_after):
            res = request_with_retry(method="GET", url=CLAUDE_TOKENIZER_REMOTE_FILE)
            res.raise_for_status()
        return Tokenizer.from_str(res.text)

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.
        :return: The responses are being returned.
        """

        human_prompt = "\n\nHuman: "
        assitant_prompt = "\n\nAssistant: "

        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )

        kwargs_with_defaults = self.model_input_kwargs

        if "stop_sequence" in kwargs:
            kwargs["stop_words"] = kwargs.pop("stop_sequence")
        if "max_tokens_to_sample" in kwargs:
            kwargs["max_length"] = kwargs.pop("max_tokens_to_sample")

        kwargs_with_defaults.update(kwargs)

        # Stream the response either in explicitly specified or if a custom handler is set
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )
        stop_words = kwargs_with_defaults.get("stop_words") or [human_prompt]

        # The human prompt must always be in the stop words list, if it's not
        # in there after the user specified some custom ones we append it
        if human_prompt not in stop_words:
            stop_words.append(human_prompt)

        # As specified by Anthropic the prompt must contain both
        # the human and assistant prompt to be valid:
        # https://console.anthropic.com/docs/prompt-design#what-is-a-prompt-
        prompt = f"{human_prompt}{prompt}{assitant_prompt}"

        data = {
            "model": self.model_name_or_path,
            "prompt": prompt,
            "max_tokens_to_sample": kwargs_with_defaults.get("max_length", self.max_length),
            "temperature": kwargs_with_defaults.get("temperature", 1),
            "top_p": kwargs_with_defaults.get("top_p", -1),
            "top_k": kwargs_with_defaults.get("top_k", -1),
            "stream": stream,
            "stop_sequences": stop_words,
        }

        if not stream:
            res = self._post(data=data)
            return [res.json()["completion"].strip()]

        res = self._post(data=data, stream=True)
        handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
        client = sseclient.SSEClient(res)
        tokens = ""
        try:
            for event in client.events():
                if event.data == TokenStreamingHandler.DONE_MARKER:
                    continue
                ed = json.loads(event.data)
                # Anthropic streamed response always includes the whole
                # string that has been streamed until that point, so
                # we can just store the last received event
                tokens = handler(ed["completion"])
        finally:
            client.close()
        return [tokens.strip()]  # return a list of strings just like non-streaming

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Make sure the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.
        :param prompt: Prompt text to be sent to the generative model.
        """
        if isinstance(prompt, List):
            raise ValueError("Anthropic invocation layer doesn't support a dictionary as prompt")

        # Tokenizer can handle truncation by itself
        token_limit = self.max_tokens_limit - self.max_length
        self.tokenizer.enable_truncation(token_limit)

        # The tokenizer we're using accepts either str or List[str],
        # if a List[str] is used we must also set is_pretokenized to True.
        # We split at spaces because if we pass the string directly the encoded prompts
        # contains strange characters in place of spaces.
        encoded_prompt: Encoding = self.tokenizer.encode(prompt.split(" "), is_pretokenized=True)

        # overflowing is the list of tokens that have been truncated
        if encoded_prompt.overflowing:
            logger.warning(
                "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
                "answer length (%s tokens) fits within the max token limit (%s tokens). "
                "Reduce the length of the prompt to prevent it from being cut off.",
                len(encoded_prompt.ids) + len(encoded_prompt.overflowing),
                self.max_tokens_limit - self.max_length,
                self.max_length,
                self.max_tokens_limit,
            )

        return " ".join(encoded_prompt.tokens)

    def _post(
        self,
        data: Dict,
        attempts: int = ANTHROPIC_MAX_RETRIES,
        status_codes_to_retry: Optional[List[int]] = None,
        timeout: float = ANTHROPIC_TIMEOUT,
        **kwargs,
    ):
        """
        Post data to Anthropic.
        Retries request in case it fails with any code in status_codes_to_retry
        or with timeout.
        All kwargs are passed to ``requests.request``, so it accepts the same arguments.
        Returns a ``requests.Response`` object.

        :param data: Object to send in the body of the request.
        :param attempts: Number of times to attempt a request in case of failures, defaults to 5.
        :param timeout: Number of seconds to wait for the server to send data before giving up, defaults to 30.
        :raises AnthropicRateLimitError: Raised if a request fails with the 429 status code.
        :raises AnthropicUnauthorizedError: Raised if a request fails with the 401 status code.
        :raises AnthropicError: Raised if requests fail for any other reason.
        :return: :class:`Response <Response>` object
        """
        if status_codes_to_retry is None:
            status_codes_to_retry = [429]

        try:
            res = request_with_retry(
                attempts=attempts,
                status_codes_to_retry=status_codes_to_retry,
                method="POST",
                url="https://api.anthropic.com/v1/complete",
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=timeout,
                **kwargs,
            )
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise AnthropicRateLimitError(f"API rate limit exceeded: {res.text}")
            if res.status_code == 401:
                raise AnthropicUnauthorizedError(f"API key is invalid: {res.text}")

            raise AnthropicError(
                f"Anthropic returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}",
                status_code=res.status_code,
            )

        return res

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Ensures Anthropic Claude Invocation Layer is selected only when Claude models are specified in
        the model name.
        """
        return model_name_or_path.startswith(("claude-v1", "claude-instant-v1"))
