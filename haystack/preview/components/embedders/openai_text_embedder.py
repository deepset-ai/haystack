from typing import List, Optional, Dict, Any
import os

import openai

from haystack.preview import component, default_to_dict


@component
class OpenAITextEmbedder:
    """
    A component for embedding strings using OpenAI models.

    Usage example:
    ```python
    from haystack.preview.components.embedders import OpenAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OpenAITextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'metadata': {'model': 'text-embedding-ada-002-v2',
    #              'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an OpenAITextEmbedder component.

        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
            environment variable OPENAI_API_KEY (recommended).
        :param model_name: The name of the OpenAI model to use. For more details on the available models,
            see [OpenAI documentation](https://platform.openai.com/docs/guides/embeddings/embedding-models).
        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details,
            see [OpenAI documentation](https://platform.openai.com/docs/api-reference/requesting-organization).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """
        # if the user does not provide the API key, check if it is set in the module client
        api_key = api_key or openai.api_key
        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "OpenAITextEmbedder expects an OpenAI API key. "
                    "Set the OPENAI_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e

        self.model_name = model_name
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix

        openai.api_key = api_key
        if organization is not None:
            openai.organization = organization

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """

        return default_to_dict(
            self, model_name=self.model_name, organization=self.organization, prefix=self.prefix, suffix=self.suffix
        )

    @component.output_types(embedding=List[float], metadata=Dict[str, Any])
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

        response = openai.Embedding.create(model=self.model_name, input=text_to_embed)

        metadata = {"model": response.model, "usage": dict(response.usage.items())}
        embedding = response.data[0]["embedding"]

        return {"embedding": embedding, "metadata": metadata}
