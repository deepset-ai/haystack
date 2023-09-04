from typing import Optional, List, Callable, Dict, Any

import logging
from dataclasses import asdict

from haystack.preview.lazy_imports import LazyImport
from haystack.preview.llm_backends.chat_message import ChatMessage
from haystack.preview.llm_backends.openai._helpers import (
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
DEFAULT_OPENAI_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 1,
    "n": 1,
    "stop": [],
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "logit_bias": {},
    "stream": False,
    "openai_organization": None,
}


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
        model_parameters: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = "https://api.openai.com/v1",
    ):
        """
        Creates an instance of ChatGPTGenerator for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param model_parameters: A dictionary of parameters to use for the model. See OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/chat) for more details. Some of the supported
            parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values means the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
            - `openai_organization`: The OpenAI organization ID.

        """
        if not api_key:
            logger.warning("OpenAI API key is missing. You will need to provide an API key to Pipeline.run().")

        self.api_key = api_key
        self.model_name = model_name
        self.model_parameters = DEFAULT_OPENAI_PARAMS | (model_parameters or {})
        self.streaming_callback = streaming_callback
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

    def complete(
        self,
        chat: List[ChatMessage],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: Optional[str] = None,
    ):
        """
        Queries the LLM with the prompts to produce replies.

        :param chat: The chat to be sent to the generative model.
        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param model_parameters: A dictionary of parameters to use for the model. See OpenAI
            [documentation](https://platform.openai.com/docs/api-reference/chat) for more details. Some of the supported
            parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values means the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
            - `openai_organization`: The OpenAI organization ID.

        """
        api_key = api_key if api_key is not None else self.api_key

        if not api_key:
            raise ValueError("OpenAI API key is missing. Please provide an API key.")

        model_name = model_name or self.model_name
        model_parameters = self.model_parameters | (model_parameters or {})
        streaming_callback = streaming_callback or self.streaming_callback
        api_base_url = api_base_url or self.api_base_url

        openai_organization = model_parameters.pop("openai_organization", None)
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
        payload = {
            "model": model_name,
            **model_parameters,
            "stream": streaming_callback is not None,
            "messages": [asdict(message) for message in chat],
        }
        if streaming_callback:
            return complete_stream(url=url, headers=headers, payload=payload, callback=streaming_callback)
        return complete(url=url, headers=headers, payload=payload)
