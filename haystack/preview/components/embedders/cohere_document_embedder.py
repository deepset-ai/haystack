from typing import List, Optional, Union, Dict, Any, Tuple
import os
from tqdm import tqdm

from haystack.preview import component, Document, default_to_dict, default_from_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    from cohere import Client, AsyncClient, CohereError

API_BASE_URL = "https://api.cohere.ai/v1/embed"


@component
class CohereDocumentEmbedder:
    """
    A component for computing Document embeddings using Cohere models.
    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "embed-english-v2.0",
        api_base_url: str = API_BASE_URL,
        truncate: str = "END",
        use_async_client: bool = False,
        max_retries: Optional[int] = 3,
        timeout: Optional[int] = 120,
        batch_size: int = 32,
        progress_bar: bool = True,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a CohereDocumentEmbedder component.

        :param api_key: The Cohere API key. It can be explicitly provided or automatically read from the environment variable COHERE_API_KEY (recommended).
        :param model_name: The name of the model to use, defaults to `"embed-english-v2.0"`. Supported Models are `"embed-english-v2.0"`/ `"large"`, `"embed-english-light-v2.0"`/ `"small"`, `"embed-multilingual-v2.0"`/ `"multilingual-22-12"`.
        :param api_base_url: The Cohere API Base url, defaults to `https://api.cohere.ai/v1/embed`.
        :param truncate: Truncate embeddings that are too long from start or end, ("NONE"|"START"|"END"), defaults to `"END"`. Passing START will discard the start of the input. END will discard the end of the input. In both cases, input is discarded until the remaining input is exactly the maximum input token length for the model. If NONE is selected, when the input exceeds the maximum input token length an error will be returned.
        :param use_async_client: Flag to select the AsyncClient, defaults to `False`. It is recommended to use AsyncClient for applications with many concurrent calls.
        :param max_retries: Maximum number of retries for requests, defaults to `3`.
        :param timeout: Request timeout in seconds, defaults to `120`.
                :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """

        if api_key is None:
            try:
                api_key = os.environ["COHERE_API_KEY"]
            except KeyError as error_msg:
                raise ValueError(
                    "CohereDocumentEmbedder expects an Cohere API key. "
                    "Please provide one by setting the environment variable COHERE_API_KEY (recommended) or by passing it explicitly."
                ) from error_msg

        self.api_key = api_key
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.truncate = truncate
        self.use_async_client = use_async_client
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            model_name=self.model_name,
            api_base_url=self.api_base_url,
            truncate=self.truncate,
            use_async_client=self.use_async_client,
            max_retries=self.max_retries,
            timeout=self.timeout,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereDocumentEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.metadata[key])
                for key in self.metadata_fields_to_embed
                if key in doc.metadata and doc.metadata[key] is not None
            ]

            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.text or ""])
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[str], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        metadata = {}
        cohere_client = Client(
            self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
        )

        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            response = cohere_client.embed(batch)
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
        if not isinstance(documents, list) or not isinstance(documents[0], Document):
            raise TypeError(
                "CohereDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the CohereTextEmbedder."
            )

        cohere_client = Client(
            self.api_key, api_url=self.api_base_url, max_retries=self.max_retries, timeout=self.timeout
        )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        all_embeddings = []
        metadata = {}
        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + self.batch_size]
            response = cohere_client.embed(batch)
            embeddings = [list(map(float, emb)) for emb in response.embeddings]
            all_embeddings.extend(embeddings)

            metadata = response.meta

        documents_with_embeddings = []
        for doc, emb in zip(documents, all_embeddings):
            doc_as_dict = doc.to_dict()
            doc_as_dict["embedding"] = emb
            documents_with_embeddings.append(Document.from_dict(doc_as_dict))

        return {"documents": documents_with_embeddings, "metadata": metadata}
