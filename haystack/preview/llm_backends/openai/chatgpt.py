from typing import Optional, List, Callable, Dict, Any

import logging
from dataclasses import asdict

from haystack.preview.lazy_imports import LazyImport
from haystack.preview.llm_backends.chat_message import ChatMessage
from haystack.preview.llm_backends.openai._helpers import (
    default_streaming_callback,
    complete,
    complete_stream,
    enforce_token_limit_chat,
    OPENAI_TOKENIZERS,
    OPENAI_TOKENIZERS_TOKEN_LIMITS,
)


with LazyImport() as tiktoken_import:
    import tiktoken


logger = logging.getLogger(__name__)


TOKENS_PER_MESSAGE_OVERHEAD = 4


class ChatGPTBackend:
    """
    ChatGPT LLM interface.

    Queries ChatGPT using OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    # TODO support function calling!

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = 500,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1,
        n: Optional[int] = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = 0,
        frequency_penalty: Optional[float] = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        stream: bool = False,
        streaming_callback: Optional[Callable] = default_streaming_callback,
        api_base_url: str = "https://api.openai.com/v1",
        openai_organization: Optional[str] = None,
    ):
        """
        Creates an instance of ChatGPTGenerator for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param max_tokens: The maximum number of tokens the output text can have.
        :param temperature: What sampling temperature to use. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model
            considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        :param n: How many completions to generate for each prompt.
        :param stop: One or more sequences where the API will stop generating further tokens.
        :param presence_penalty: What penalty to apply if a token is already present at all. Bigger values mean
            the model will be less likely to repeat the same token in the text.
        :param frequency_penalty: What penalty to apply if a token has already been generated in the text.
            Bigger values mean the model will be less likely to repeat the same token in the text.
        :param logit_bias: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
            values are the bias to add to that token.
        :param stream: If set to True, the API will stream the response. The streaming_callback parameter
            is used to process the stream. If set to False, the response will be returned as a string.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param openai_organization: The OpenAI organization ID.

        See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
        """
        if not api_key:
            logger.warning("OpenAI API key is missing. You will need to provide an API key to Pipeline.run().")

        self.api_key = api_key
        self.model_name = model_name

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop or []
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias or {}
        self.stream = stream
        self.streaming_callback = streaming_callback or default_streaming_callback

        self.openai_organization = openai_organization
        self.api_base_url = api_base_url

        tokenizer = None
        for model_prefix, tokenizer_name in OPENAI_TOKENIZERS.items():
            if model_name.startswith(model_prefix):
                tokenizer = tiktoken.get_encoding(tokenizer_name)
                break
        if not tokenizer:
            raise ValueError(f"Tokenizer for model '{model_name}' not found.")
        self.tokenizer = tokenizer

        max_tokens_limit = None
        for model_prefix, limit in OPENAI_TOKENIZERS_TOKEN_LIMITS.items():
            if model_name.startswith(model_prefix):
                max_tokens_limit = limit
                break
        if not max_tokens_limit:
            raise ValueError(f"Max tokens limit for model '{model_name}' not found.")
        self.max_tokens_limit = max_tokens_limit

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a dictionary.
        """
        return {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "stream": self.stream,
            # FIXME how to serialize the streaming callback?
            "api_base_url": self.api_base_url,
            "openai_organization": self.openai_organization,
        }

    def complete(
        self,
        chat: List[ChatMessage],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        api_base_url: Optional[str] = None,
        openai_organization: Optional[str] = None,
        stream: Optional[bool] = None,
        streaming_callback: Optional[Callable] = None,
    ):
        """
        Queries the LLM with the prompts to produce replies.

        :param chat: The chat to be sent to the generative model.
        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param max_tokens: The maximum number of tokens the output text can have.
        :param temperature: What sampling temperature to use. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model
            considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        :param n: How many completions to generate for each prompt.
        :param stop: One or more sequences where the API will stop generating further tokens.
        :param presence_penalty: What penalty to apply if a token is already present at all. Bigger values mean
            the model will be less likely to repeat the same token in the text.
        :param frequency_penalty: What penalty to apply if a token has already been generated in the text.
            Bigger values mean the model will be less likely to repeat the same token in the text.
        :param logit_bias: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
            values are the bias to add to that token.
        :param stream: If set to True, the API will stream the response. The streaming_callback parameter
            is used to process the stream. If set to False, the response will be returned as a string.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param openai_organization: The OpenAI organization ID.

        See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
        """
        api_key = api_key if api_key is not None else self.api_key

        if not api_key:
            raise ValueError("OpenAI API key is missing. Please provide an API key.")

        model_name = model_name if model_name is not None else self.model_name
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        n = n if n is not None else self.n
        stop = stop if stop is not None else self.stop
        presence_penalty = presence_penalty if presence_penalty is not None else self.presence_penalty
        frequency_penalty = frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        logit_bias = logit_bias if logit_bias is not None else self.logit_bias
        stream = stream if stream is not None else self.stream
        streaming_callback = streaming_callback if streaming_callback is not None else self.streaming_callback
        api_base_url = api_base_url or self.api_base_url
        openai_organization = openai_organization if openai_organization is not None else self.openai_organization

        parameters = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if openai_organization:
            headers["OpenAI-Organization"] = openai_organization
        url = f"{api_base_url}/chat/completions"

        chat = enforce_token_limit_chat(
            chat=chat,
            tokenizer=self.tokenizer,
            max_tokens_limit=self.max_tokens_limit,
            tokens_per_message_overhead=TOKENS_PER_MESSAGE_OVERHEAD,
        )
        payload = {**parameters, "messages": [asdict(message) for message in chat]}
        if stream:
            return complete_stream(url=url, headers=headers, payload=payload, callback=streaming_callback)
        else:
            return complete(url=url, headers=headers, payload=payload)
