from typing import List, Optional, Dict, Any, Tuple
import os

import openai
from tqdm import tqdm


from haystack.preview import component, Document, default_to_dict


@component
class OpenAIDocumentEmbedder:
    """
    A component for computing Document embeddings using OpenAI models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.preview import Document
    from haystack.preview.components.embedders import OpenAIDocumentEmbedder

    doc = Document(text="I love pizza!")

    document_embedder = OpenAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OpenAIDocumentEmbedder component.
        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
                        environment variable OPENAI_API_KEY (recommended).
        :param model_name: The name of the model to use.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param organization: The OpenAI-Organization ID, defaults to `None`. For more details, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        # if the user does not provide the API key, check if it is set in the module client
        api_key = api_key or openai.api_key
        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "OpenAIDocumentEmbedder expects an OpenAI API key. "
                    "Set the OPENAI_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e

        self.model_name = model_name
        self.organization = organization
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

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
            self,
            model_name=self.model_name,
            organization=self.organization,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.metadata_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

            # copied from OpenAI embedding_utils (https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py)
            # replace newlines, which can negatively affect performance.
            text_to_embed = text_to_embed.replace("\n", " ")
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        metadata = {}
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            response = openai.Embedding.create(model=self.model_name, input=batch)
            embeddings = [el["embedding"] for el in response.data]
            all_embeddings.extend(embeddings)

            if "model" not in metadata:
                metadata["model"] = response.model
            if "usage" not in metadata:
                metadata["usage"] = dict(response.usage.items())
            else:
                metadata["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                metadata["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, metadata

    @component.output_types(documents=List[Document], metadata=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "OpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OpenAITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, metadata = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "metadata": metadata}
