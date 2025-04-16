# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, Optional

from openai.lib.azure import AsyncAzureOpenAI, AzureADTokenProvider, AzureOpenAI

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.http_client import init_http_client


@component
class AzureOpenAITextEmbedder(OpenAITextEmbedder):
    """
    Embeds strings using OpenAI models deployed on Azure.

    ### Usage example

    ```python
    from haystack.components.embedders import AzureOpenAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = AzureOpenAITextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'text-embedding-ada-002-v2',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    # pylint: disable=super-init-not-called
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = "2023-05-15",
        azure_deployment: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_key: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False),
        azure_ad_token: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_AD_TOKEN", strict=False),
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        prefix: str = "",
        suffix: str = "",
        *,
        default_headers: Optional[Dict[str, str]] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an AzureOpenAITextEmbedder component.

        :param azure_endpoint:
            The endpoint of the model deployed on Azure.
        :param api_version:
            The version of the API to use.
        :param azure_deployment:
            The name of the model deployed on Azure. The default model is text-embedding-ada-002.
        :param dimensions:
            The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3
            and later models.
        :param api_key:
            The Azure OpenAI API key.
            You can set it with an environment variable `AZURE_OPENAI_API_KEY`, or pass with this
            parameter during initialization.
        :param azure_ad_token:
            Microsoft Entra ID token, see Microsoft's
            [Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
            documentation for more information. You can set it with an environment variable
            `AZURE_OPENAI_AD_TOKEN`, or pass with this parameter during initialization.
            Previously called Azure Active Directory.
        :param organization:
            Your organization ID. See OpenAI's
            [Setting Up Your Organization](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param timeout: The timeout for `AzureOpenAI` client calls, in seconds.
            If not set, defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries: Maximum number of retries to contact AzureOpenAI after an internal error.
            If not set, defaults to either the `OPENAI_MAX_RETRIES` environment variable, or to 5 retries.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param default_headers: Default headers to send to the AzureOpenAI client.
        :param azure_ad_token_provider: A function that returns an Azure Active Directory token, will be invoked on
            every request.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

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

        self.api_key = api_key  # type: ignore[assignment] # mypy does not understand that api_key can be None
        self.azure_ad_token = azure_ad_token
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.model = azure_deployment
        self.dimensions = dimensions
        self.organization = organization
        self.timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        self.prefix = prefix
        self.suffix = suffix
        self.default_headers = default_headers or {}
        self.azure_ad_token_provider = azure_ad_token_provider
        self.http_client_kwargs = http_client_kwargs

        client_kwargs: Dict[str, Any] = {
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "azure_ad_token_provider": azure_ad_token_provider,
            "api_key": api_key.resolve_value() if api_key is not None else None,
            "azure_ad_token": azure_ad_token.resolve_value() if azure_ad_token is not None else None,
            "organization": organization,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
        }

        self.client = AzureOpenAI(
            http_client=init_http_client(self.http_client_kwargs, async_client=False), **client_kwargs
        )
        self.async_client = AsyncAzureOpenAI(
            http_client=init_http_client(self.http_client_kwargs, async_client=True), **client_kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        azure_ad_token_provider_name = None
        if self.azure_ad_token_provider:
            azure_ad_token_provider_name = serialize_callable(self.azure_ad_token_provider)
        return default_to_dict(
            self,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            dimensions=self.dimensions,
            organization=self.organization,
            api_version=self.api_version,
            prefix=self.prefix,
            suffix=self.suffix,
            api_key=self.api_key.to_dict() if self.api_key is not None else None,
            azure_ad_token=self.azure_ad_token.to_dict() if self.azure_ad_token is not None else None,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            azure_ad_token_provider=azure_ad_token_provider_name,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOpenAITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "azure_ad_token"])
        serialized_azure_ad_token_provider = data["init_parameters"].get("azure_ad_token_provider")
        if serialized_azure_ad_token_provider:
            data["init_parameters"]["azure_ad_token_provider"] = deserialize_callable(
                serialized_azure_ad_token_provider
            )
        return default_from_dict(cls, data)
