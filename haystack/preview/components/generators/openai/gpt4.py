from typing import Optional, Callable

import logging

from haystack.preview import component
from haystack.preview.components.generators.openai.gpt35 import GPT35Generator, API_BASE_URL


logger = logging.getLogger(__name__)


@component
class GPT4Generator(GPT35Generator):
    """
    LLM Generator compatible with GPT4 large language models.

    Queries the LLM using OpenAI's API. Invocations are made using OpenAI SDK ('openai' package)
    See [OpenAI GPT4 API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Creates an instance of GPT4Generator for OpenAI's GPT-4 model.

        :param api_key: The OpenAI API key.
        :param model_name: The name of the model to use.
        :param system_prompt: An additional message to be sent to the LLM at the beginning of each conversation.
            Typically, a conversation is formatted with a system message first, followed by alternating messages from
            the 'user' (the "queries") and the 'assistant' (the "responses"). The system message helps set the behavior
            of the assistant. For example, you can modify the personality of the assistant or provide specific
            instructions about how it should behave throughout the conversation.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the OpenAI
            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
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
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            **kwargs,
        )
