from typing import Optional, List, Callable, Dict, Any

import logging

from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.llm_backends.openai.chatgpt import ChatGPTBackend
from haystack.preview.llm_backends.chat_message import ChatMessage
from haystack.preview.llm_backends.openai._helpers import default_streaming_callback


logger = logging.getLogger(__name__)


TOKENS_PER_MESSAGE_OVERHEAD = 4


@component
class ChatGPTGenerator:
    """
    ChatGPT LLM Generator.

    Queries ChatGPT using OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    # TODO support function calling!

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
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
        :param system_prompt: The prompt to be prepended to the user prompt.
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
        self.llm = ChatGPTBackend(
            api_key=api_key,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            stream=stream,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            openai_organization=openai_organization,
        )
        self.system_prompt = system_prompt

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, system_prompt=self.system_prompt, **self.llm.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatGPTGenerator":
        """
        Deserialize this component from a dictionary.
        """
        # FIXME how to deserialize the streaming callback?
        return default_from_dict(cls, data)

    @component.output_types(replies=List[List[str]], metadata=List[Dict[str, Any]])
    def run(
        self,
        prompts: List[str],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        stream: Optional[bool] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: Optional[str] = None,
        openai_organization: Optional[str] = None,
    ):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompts: The prompts to be sent to the generative model.
        :param api_key: The OpenAI API key.
        :param model_name: The name or path of the underlying model.
        :param system_prompt: The prompt to be prepended to the user prompt.
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
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        if system_prompt:
            system_message = ChatMessage(content=system_prompt, role="system")
        chats = []
        for prompt in prompts:
            message = ChatMessage(content=prompt, role="user")
            if system_prompt:
                chats.append([system_message, message])
            else:
                chats.append([message])

        replies, metadata = [], []
        for chat in chats:
            reply, meta = self.llm.complete(
                chat=chat,
                api_key=api_key,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                api_base_url=api_base_url,
                openai_organization=openai_organization,
                stream=stream,
                streaming_callback=streaming_callback,
            )
            replies.append(reply)
            metadata.append(meta)

        return {"replies": replies, "metadata": metadata}
