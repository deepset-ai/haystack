# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Optional

from openai.lib.azure import AzureOpenAI

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class AzureOpenAITextEmbedder:
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

    def __init__(
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
        """
        # Why is this here?
        # AzureOpenAI init is forcing us to use an init method that takes either base_url or azure_endpoint as not
        # None init parameters. This way we accommodate the use case where env var AZURE_OPENAI_ENDPOINT is set instead
        # of passing it as a parameter.
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("Please provide an Azure endpoint or set the environment variable AZURE_OPENAI_ENDPOINT.")

        if api_key is None and azure_ad_token is None:
            raise ValueError("Please provide an API key or an Azure Active Directory token.")

        self.api_key = api_key
        self.azure_ad_token = azure_ad_token
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.dimensions = dimensions
        self.organization = organization
        self.timeout = timeout or float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        self.max_retries = max_retries or int(os.environ.get("OPENAI_MAX_RETRIES", 5))
        self.prefix = prefix
        self.suffix = suffix

        self._client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_key=api_key.resolve_value() if api_key is not None else None,
            azure_ad_token=azure_ad_token.resolve_value() if azure_ad_token is not None else None,
            organization=organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.azure_deployment}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
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
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """
        Embeds a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(text, str):
            # Check if input is a list and all elements are instances of Document
            if isinstance(text, list) and all(isinstance(elem, Document) for elem in text):
                error_message = "Input must be a string. Use AzureOpenAIDocumentEmbedder for a list of Documents."
            else:
                error_message = "Input must be a string."
            raise TypeError(error_message)

        # Preprocess the text by adding prefixes/suffixes
        # finally, replace newlines as recommended by OpenAI docs
        processed_text = f"{self.prefix}{text}{self.suffix}".replace("\n", " ")

        if self.dimensions is not None:
            response = self._client.embeddings.create(
                model=self.azure_deployment, dimensions=self.dimensions, input=processed_text
            )
        else:
            response = self._client.embeddings.create(model=self.azure_deployment, input=processed_text)

        return {
            "embedding": response.data[0].embedding,
            "meta": {"model": response.model, "usage": dict(response.usage)},
        }
