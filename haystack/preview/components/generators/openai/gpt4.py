from typing import Optional, List, Callable, Dict

import logging

from haystack.preview import component
from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator, default_streaming_callback


logger = logging.getLogger(__name__)


@component
class GPT4Generator(ChatGPTGenerator):
    """
    GPT4 LLM Generator.

    Queries GPT4 using OpenAI's GPT-3 ChatGPT API. Invocations are made using REST API.
    See [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
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
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            max_reply_tokens=max_reply_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            moderate_content=moderate_content,
            stream=stream,
            streaming_callback=streaming_callback,
            streaming_done_marker=streaming_done_marker,
            api_base_url=api_base_url,
            openai_organization=openai_organization,
        )
