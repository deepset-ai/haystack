# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Callable, Dict, Optional

# pylint: disable=import-error
from openai.lib.azure import AzureOpenAI

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


@component
class AzureOpenAIChatGenerator(OpenAIChatGenerator):
    """
    A Chat Generator component that uses the Azure OpenAI API to generate text.

    Enables text generation using OpenAI's large language models (LLMs) on Azure. It supports `gpt-4` and `gpt-3.5-turbo`
    family of models accessed through the chat completions API endpoint.

    Users can pass any text generation parameters valid for the `openai.ChatCompletion.create` method
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    For more details on OpenAI models deployed on Azure, refer to the Microsoft
    [documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/).

    Key Features and Compatibility:
     - Primary Compatibility: Designed to work seamlessly with the OpenAI API Chat Completion endpoint.
     - Streaming Support: Supports streaming responses from the OpenAI API Chat Completion endpoint.
     - Customizability: Supports all parameters supported by the OpenAI API Chat Completion endpoint.

    Input and Output Format:
      - ChatMessage Format: This component uses the ChatMessage format for structuring both input and output, ensuring
        coherent and contextually relevant responses in chat-based text generation scenarios.
     - Details on the ChatMessage format can be found [here](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage).


    Usage example:

    ```python
    from haystack.components.generators.chat import AzureOpenAIGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = AzureOpenAIGenerator(
        azure_endpoint="<Your Azure endpoint e.g. `https://your-company.azure.openai.com/>",
        api_key=Secret.from_token("<your-api-key>"),
        azure_deployment="<this a model name, e.g. gpt-35-turbo>")
    response = client.run(messages)
    print(response)
    ```

    ```
    {'replies':
        [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
         enabling computers to understand, interpret, and generate human language in a way that is meaningful and useful.',
         role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
         meta={'model': 'gpt-3.5-turbo-0613', 'index': 0, 'finish_reason': 'stop',
         'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]
    }
    ```
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = "2023-05-15",
        azure_deployment: Optional[str] = "gpt-35-turbo",
        api_key: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False),
        azure_ad_token: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_AD_TOKEN", strict=False),
        organization: Optional[str] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        timeout: Optional[float] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Azure OpenAI Chat Generator component.

        :param azure_endpoint: The endpoint of the deployed model, e.g. `"https://example-resource.azure.openai.com/"`
        :param api_version: The version of the API to use. Defaults to 2023-05-15
        :param azure_deployment: The deployment of the model, usually the model name.
        :param api_key: The API key to use for authentication.
        :param azure_ad_token: [Azure Active Directory token](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
        :param organization: The Organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
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
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        """
        # We intentionally do not call super().__init__ here because we only need to instantiate the client to interact
        # with the API.

        # Why is this here?
        # AzureOpenAI init is forcing us to use an init method that takes either base_url or azure_endpoint as not
        # None init parameters. This way we accommodate the use case where env var AZURE_OPENAI_ENDPOINT is set instead
        # of passing it as a parameter.
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("Please provide an Azure endpoint or set the environment variable AZURE_OPENAI_ENDPOINT.")

        if api_key is None and azure_ad_token is None:
            raise ValueError("Please provide an API key or an Azure Active Directory token.")

        # The check above makes mypy incorrectly infer that api_key is never None,
        # which propagates the incorrect type.
        self.api_key = api_key  # type: ignore
        self.azure_ad_token = azure_ad_token
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.organization = organization
        self.model = azure_deployment or "gpt-35-turbo"
        self.timeout = timeout

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_key=api_key.resolve_value() if api_key is not None else None,
            azure_ad_token=azure_ad_token.resolve_value() if azure_ad_token is not None else None,
            organization=organization,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            organization=self.organization,
            api_version=self.api_version,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            api_key=self.api_key.to_dict() if self.api_key is not None else None,
            azure_ad_token=self.azure_ad_token.to_dict() if self.azure_ad_token is not None else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOpenAIChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "azure_ad_token"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)
