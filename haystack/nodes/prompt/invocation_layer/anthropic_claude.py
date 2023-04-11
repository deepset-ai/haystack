import os
from typing import Dict, List, Union, Optional, cast
import json
import logging

import sseclient
from transformers import GPT2TokenizerFast

from haystack.errors import AnthropicError, AnthropicRateLimitError, AnthropicUnauthorizedError
from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.utils.requests import request_with_retry
from haystack.utils.openai_utils import USE_TIKTOKEN
from haystack.environment import HAYSTACK_REMOTE_API_MAX_RETRIES, HAYSTACK_REMOTE_API_TIMEOUT_SEC

ANTHROPIC_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
ANTHROPIC_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))

logger = logging.getLogger(__name__)


class AnthropicClaudeInvocationLayer(PromptModelInvocationLayer):
    """
    Anthropic Claude Invocation Layer
    This layer is used to invoke the Claude API provided by Anthropic.
    """

    def __init__(
        self, api_key: str, model_name_or_path: str = "claude-v1", max_tokens_to_sample: Optional[int] = 200, **kwargs
    ):
        """
         Creates an instance of PromptModelInvocation Layer for Anthropic's Claude models.
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
        self.max_tokens_to_sample = max_tokens_to_sample

        # 200 is the default length for answers from Anthropic
        # max_tokens_to_sample must be set otherwise AnthropicInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_tokens_to_sample or 200

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to Anthropic documentation
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "max_tokens_to_sample",
                "temperature",
                "top_p",
                "top_k",
                "stop_sequences",
                "stream",
                "stream_handler",
            ]
            if key in kwargs
        }

        (tokenizer_name, max_tokens_limit) = _anthropic_text_completion_tokenization_details(
            model_name=self.model_name_or_path
        )
        self.max_tokens_limit = max_tokens_limit
        self._tokenizer = load_anthropic_tokenizer(tokenizer_name=tokenizer_name)

    @property
    def url(self) -> str:
        return "https://api.anthropic.com/v1/complete"

    @property
    def headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "Content-Type": "application/json"}

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

        # we use keyword stop_words but Anthropic uses stop_sequences
        if "stop_words" in kwargs:
            kwargs["stop_sequences"] = kwargs.pop("stop_words")
        if "max_tokens" in kwargs:
            kwargs["max_tokens_to_sample"] = kwargs.pop("max_tokens")
        kwargs_with_defaults.update(kwargs)

        # either stream is True (will use default handler) or stream_handler is provided
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )
        stop_sequences = kwargs.get("stop_sequences", ["\n\nHuman: "])

        data = {
            "model": self.model_name_or_path,
            "prompt": "{} {} {}".format(human_prompt, prompt, assitant_prompt),
            "max_tokens_to_sample": kwargs_with_defaults.get("max_tokens_to_sample", self.max_tokens_to_sample),
            "temperature": kwargs_with_defaults.get("temperature", 1),
            "top_p": kwargs_with_defaults.get("top_p", -1),
            "top_k": kwargs_with_defaults.get("top_k", -1),
            "stream": stream,
            "stop_sequences": stop_sequences,
        }

        if not stream:
            res = self._post(data=data)
            return [res.json()["completion"].strip()]

        res = self._post(data=data, stream=True)
        handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
        client = sseclient.SSEClient(res)
        tokens: List[str] = []
        try:
            for event in client.events():
                if event.data != TokenStreamingHandler.DONE_MARKER:
                    ed = json.loads(event.data)
                    token: str = ed["choices"][0]["text"]
                    tokens.append(handler(token, event_data=ed["completion"]))
        finally:
            client.close()
        return ["".join(tokens)]  # return a list of strings just like non-streaming

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.
        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = count_anthropic_tokens(cast(str, prompt), self._tokenizer)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )

        if USE_TIKTOKEN:
            tokenized_payload = self._tokenizer.encode(prompt)
            decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_tokens_limit - n_answer_tokens])
        else:
            tokenized_payload = self._tokenizer.tokenize(prompt)
            decoded_string = self._tokenizer.convert_tokens_to_string(
                tokenized_payload[: self.max_tokens_limit - n_answer_tokens]
            )
        return decoded_string

    def _post(
        self,
        data: Dict,
        attempts: int = ANTHROPIC_MAX_RETRIES,
        status_codes: Optional[List[int]] = None,
        timeout: float = ANTHROPIC_TIMEOUT,
        **kwargs,
    ):
        """
        Post data to Anthropic using this invocation layer url and headers.
        Retries request in case it fails with any code in status_codes
        or with timeout.
        All kwargs will be passed to ``requests.request``, so it accepts the same arguments.
        Returns a ``requests.Response`` object.

        :param data: Object to send in the body of the request
        :param attempts: Number of times to attempt request in case of failures, defaults to 5
        :param timeout: How many seconds to wait for the server to send data before giving up, defaults to 30
        :raises AnthropicRateLimitError: Raised if requests fails with 429 status code
        :raises AnthropicUnauthorizedError: Raised if requests fail with 401 status code
        :raises AnthropicError: Raised if requests fail for any other reason
        :return: :class:`Response <Response>` object
        """
        if status_codes is None:
            status_codes = [429]

        res = request_with_retry(
            attempts=attempts,
            status_codes=status_codes,
            method="POST",
            url=self.url,
            headers=self.headers,
            data=data,
            timeout=timeout,
            **kwargs,
        )

        if res.status_code == 429:
            raise AnthropicRateLimitError(f"API rate limit exceeded: {res.text}")
        if res.status_code == 401:
            raise AnthropicUnauthorizedError(f"API key is invalid: {res.text}")
        if not res.ok:
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
        valid_model = any(
            m
            for m in ["claude-v1", "claude-v1.0", "claude-v1.2", "claude-instant-v1", "claude-instant-v1.0"]
            if m in model_name_or_path
        )
        return valid_model


def load_anthropic_tokenizer(tokenizer_name: str):
    """Load the GPT2TokenizerFast from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    """
    logger.debug("Using GPT2TokenizerFast tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_anthropic_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in `text` based on the `tokenizer`.

    :param text: A string to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    """
    return len(tokenizer.tokenize(text))


def count_anthropic_tokens_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count the number of tokens in `messages` based on the `tokenizer` provided.

    :param messages: The messages to be tokenized.
    :param tokenizer: An tokenizer.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(tokenizer.tokenize(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def _anthropic_text_completion_tokenization_details(model_name: str):
    """Return the tokenizer name and max tokens limit for a given Anthropic `model_name`.

    :param model_name: Name of the Anthropic model.
    """
    tokenizer_name = "gpt2"
    max_tokens_limit = (
        9000  # Based on this ref: https://cdn2.assets-servd.host/anthropic-website/production/images/FINAL-PRICING.pdf
    )

    return tokenizer_name, max_tokens_limit
