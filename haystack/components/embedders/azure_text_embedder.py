import os
from typing import Any, Dict, List, Optional

from openai.lib.azure import AzureOpenAI

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class AzureOpenAITextEmbedder:
    """
    A component for embedding strings using OpenAI models on Azure.

    Usage example:
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
        api_key: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False),
        azure_ad_token: Optional[Secret] = Secret.from_env_var("AZURE_OPENAI_AD_TOKEN", strict=False),
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an AzureOpenAITextEmbedder component.

        :param azure_endpoint:
            The endpoint of the deployed model.
        :param api_version:
            The version of the API to use.
        :param azure_deployment:
            The deployment of the model, usually matches the model name.
        :param api_key:
            The API key used for authentication.
        :param azure_ad_token:
            Microsoft Entra ID token, see Microsoft's official
            [Entra ID](https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id)
            documentation for more information.
            Used to be called Azure Active Directory.
        :param organization:
            The Organization ID. See OpenAI's
            [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
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
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix

        self._client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_key=api_key.resolve_value() if api_key is not None else None,
            azure_ad_token=azure_ad_token.resolve_value() if azure_ad_token is not None else None,
            organization=organization,
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
            organization=self.organization,
            api_version=self.api_version,
            prefix=self.prefix,
            suffix=self.suffix,
            api_key=self.api_key.to_dict() if self.api_key is not None else None,
            azure_ad_token=self.azure_ad_token.to_dict() if self.azure_ad_token is not None else None,
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
        Embed a single string.

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

        response = self._client.embeddings.create(model=self.azure_deployment, input=processed_text)

        return {
            "embedding": response.data[0].embedding,
            "meta": {"model": response.model, "usage": dict(response.usage)},
        }
