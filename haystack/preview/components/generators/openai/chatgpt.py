from typing import Optional, List, Callable, Dict, Any

import logging

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.components.generators._helpers import enforce_token_limit
from haystack.preview.components.generators.openai._helpers import (
    default_streaming_callback,
    query_chat_model,
    query_chat_model_stream,
    TOKENIZERS,
    TOKENIZERS_TOKEN_LIMITS,
)


with LazyImport() as tiktoken_import:
    import tiktoken


logger = logging.getLogger(__name__)


@component
class ChatGPTGenerator:
    """
    ChatGPT LLM Generator.

    Queries ChatGPT using OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        max_reply_tokens: Optional[int] = 500,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1,
        n: Optional[int] = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = 0,
        frequency_penalty: Optional[float] = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        moderate_content: bool = True,
        stream: bool = False,
        streaming_callback: Optional[Callable] = default_streaming_callback,
        streaming_done_marker="[DONE]",
        api_base_url: str = "https://api.openai.com/v1",
        openai_organization: Optional[str] = None,
    ):
        """
        Creates an instance of ChatGPTGenerator for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param system_prompt: The prompt to be prepended to the user prompt.
        :param max_reply_tokens: The maximum number of tokens the output text can have.
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
        :param moderate_content: If set to True, the input and generated answers are filtered for potentially
            sensitive content using the [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation).
            If the input or answers are flagged, an empty list is returned in place of the answers.
        :param stream: If set to True, the API will stream the response. The streaming_callback parameter
            is used to process the stream. If set to False, the response will be returned as a string.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param streaming_done_marker: A marker that indicates the end of the stream. The marker is used to determine
            when to stop streaming. Defaults to "[DONE]".
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param openai_organization: The OpenAI organization ID.

        See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
        """
        if not api_key:
            logger.warning("OpenAI API key is missing. You will need to provide an API key to Pipeline.run().")

        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.max_reply_tokens = max_reply_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.moderate_content = moderate_content
        self.stream = stream
        self.streaming_callback = streaming_callback
        self.streaming_done_marker = streaming_done_marker

        self.openai_organization = openai_organization
        self.api_base_url = api_base_url

        self.tokenizer = None
        for model_prefix in TOKENIZERS:
            if model_name.startswith(model_prefix):
                self.tokenizer = tiktoken.get_encoding(TOKENIZERS[model_prefix])
                break
        if not self.tokenizer:
            raise ValueError(f"Tokenizer for model {model_name} not found.")

        self.max_tokens_limit = None
        for model_prefix in TOKENIZERS_TOKEN_LIMITS:
            if model_name.startswith(model_prefix):
                self.max_tokens_limit = TOKENIZERS_TOKEN_LIMITS[model_prefix]
                break
        if not self.max_tokens_limit:
            raise ValueError(f"Max tokens limit for model {model_name} not found.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            max_reply_tokens=self.max_reply_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            moderate_content=self.moderate_content,
            stream=self.stream,
            # FIXME how to serialize the streaming callback?
            streaming_done_marker=self.streaming_done_marker,
            api_base_url=self.api_base_url,
            openai_organization=self.openai_organization,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatGPTGenerator":
        """
        Deserialize this component from a dictionary.
        """
        # FIXME how to deserialize the streaming callback?
        return default_from_dict(cls, data)

    @component.output_types(replies=List[List[str]])
    def run(
        self,
        prompts: List[str],
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        max_reply_tokens: Optional[int] = 500,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 1,
        n: Optional[int] = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = 0,
        frequency_penalty: Optional[float] = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        moderate_content: bool = True,
        api_base_url: str = "https://api.openai.com/v1",
        openai_organization: Optional[str] = None,
        stream: bool = False,
        streaming_callback: Optional[Callable] = None,
        streaming_done_marker: str = "[DONE]",
    ):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompts: The prompts to be sent to the generative model.
        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param system_prompt: The prompt to be prepended to the user prompt.
        :param max_reply_tokens: The maximum number of tokens the output text can have.
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
        :param moderate_content: If set to True, the input and generated answers are filtered for potentially
            sensitive content using the [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation).
            If the input or answers are flagged, an empty list is returned in place of the answers.
        :param stream: If set to True, the API will stream the response. The streaming_callback parameter
            is used to process the stream. If set to False, the response will be returned as a string.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param streaming_done_marker: A marker that indicates the end of the stream. The marker is used to determine
            when to stop streaming. Defaults to "[DONE]".
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param openai_organization: The OpenAI organization ID.

        See OpenAI documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
        """
        if not api_key and not self.api_key:
            raise ValueError("OpenAI API key is missing. Please provide an API key.")

        stream = stream or self.stream
        parameters = {
            "model": model_name or self.model_name,
            "max_reply_tokens": max_reply_tokens or self.max_reply_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "n": n or self.n,
            "stream": stream,
            "stop": stop or self.stop,
            "presence_penalty": presence_penalty or self.presence_penalty,
            "frequency_penalty": frequency_penalty or self.frequency_penalty,
            "logit_bias": logit_bias or self.logit_bias,
            "moderate_content": moderate_content or self.moderate_content,
        }

        headers = {"Authorization": f"Bearer {api_key or self.api_key}", "Content-Type": "application/json"}
        if openai_organization or self.openai_organization:
            headers["OpenAI-Organization"] = openai_organization or self.openai_organization

        url = f"{api_base_url or self.api_base_url}/chat/completions"

        replies = []
        streaming_callback = streaming_callback or self.streaming_callback
        for prompt in prompts:
            payload = {
                **parameters,
                "messages": enforce_token_limit(
                    prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    tokenizer=self.tokenizer,
                    max_tokens_limit=self.max_tokens_limit,
                ),
            }
            if stream:
                reply = query_chat_model_stream(
                    url=url, headers=headers, payload=payload, callback=streaming_callback, marker=streaming_done_marker
                )
            else:
                reply = query_chat_model(url=url, headers=headers, payload=payload)
            replies.append(reply)

        return {"replies": replies}
