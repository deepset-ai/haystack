from typing import Any, Dict, List, Optional

from openai import OpenAI

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class OpenAITextEmbedder:
    """
    A component for embedding strings using OpenAI models.

    Usage example:
    ```python
    from haystack.components.embedders import OpenAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OpenAITextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'text-embedding-ada-002-v2',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "text-embedding-ada-002",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an OpenAITextEmbedder component.

        :param api_key:
            The OpenAI API key.
        :param model:
            The name of the model to use.
        :param dimensions:
            The number of dimensions the resulting output embeddings should have. Only supported in `text-embedding-3` and later models.
        :param api_base_url:
            Overrides default base url for all HTTP requests.
        :param organization:
            The Organization ID. See OpenAI's
            [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization)
            for more information.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        """
        self.model = model
        self.dimensions = dimensions
        self.api_base_url = api_base_url
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.api_key = api_key

        self.client = OpenAI(api_key=api_key.resolve_value(), organization=organization, base_url=api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_base_url=self.api_base_url,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            dimensions=self.dimensions,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
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
            raise TypeError(
                "OpenAITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the OpenAIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
        # replace newlines, which can negatively affect performance.
        text_to_embed = text_to_embed.replace("\n", " ")

        if self.dimensions is not None:
            response = self.client.embeddings.create(model=self.model, dimensions=self.dimensions, input=text_to_embed)
        else:
            response = self.client.embeddings.create(model=self.model, input=text_to_embed)

        meta = {"model": response.model, "usage": dict(response.usage)}

        return {"embedding": response.data[0].embedding, "meta": meta}
