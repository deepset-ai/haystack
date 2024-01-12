from typing import List, Optional, Dict, Any

from openai import OpenAI

from haystack import component, default_to_dict


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
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an OpenAITextEmbedder component.

        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
            environment variable OPENAI_API_KEY (recommended).
        :param model: The name of the OpenAI model to use. For more details on the available models,
            see [OpenAI documentation](https://platform.openai.com/docs/guides/embeddings/embedding-models).
        :param organization: The Organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param api_base_url: The OpenAI API Base url, defaults to None. For more details, see OpenAI [docs](https://platform.openai.com/docs/api-reference/audio).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """
        self.model = model
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix

        self.client = OpenAI(api_key=api_key, organization=organization, base_url=api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """

        return default_to_dict(
            self, model=self.model, organization=self.organization, prefix=self.prefix, suffix=self.suffix
        )

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "OpenAITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the OpenAIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
        # replace newlines, which can negatively affect performance.
        text_to_embed = text_to_embed.replace("\n", " ")

        response = self.client.embeddings.create(model=self.model, input=text_to_embed)
        meta = {"model": response.model, "usage": dict(response.usage)}

        return {"embedding": response.data[0].embedding, "meta": meta}
