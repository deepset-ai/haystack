from typing import List, Optional, Dict, Any
import os

import requests
from tqdm import tqdm

from haystack import component, Document, default_to_dict

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"

@component
class JinaDocumentEmbedder:
    """
    A component for computing Document embeddings using Jina AI models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.embedders import JinaDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = JinaDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "jina-embeddings-v2-base-en",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a JinaDocumentEmbedder component.
        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        :param model_name: The name of the Jina model to use.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        # if the user does not provide the API key, check if it is set in the module client
        if api_key is None:
            try:
                api_key = os.environ["JINA_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "JinaDocumentEmbedder expects an Jina API key. "
                    "Set the JINA_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e

        self.model_name = model_name
        self.prefix = prefix
        self.suffix = suffix
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

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
            response = self._session.post(  # type: ignore
                JINA_API_URL, json={"input": batch, "model": self._model_name}
            ).json()
            if "data" not in response:
                raise RuntimeError(response["detail"])

            # Sort resulting embeddings by index
            sorted_embeddings = sorted(response["data"], key=lambda e: e["index"])  # type: ignore
            embeddings = [result["embedding"] for result in sorted_embeddings]
            all_embeddings.extend(embeddings)

            if "model" not in metadata:
                metadata["model"] = response["model"]
            if "usage" not in metadata:
                metadata["usage"] = dict(response["usage"].items())
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
                "JinaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the JinaTextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, metadata = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "metadata": metadata}
