import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tqdm import tqdm

from haystack import component, default_to_dict
from haystack.components.embedders.hf_utils import check_valid_model
from haystack.dataclasses import Document
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@component
class HuggingFaceTEIDocumentEmbedder:
    """
    A component for computing Document embeddings using HuggingFace Text-Embeddings-Inference endpoints. This component
    is designed to seamlessly inference models deployed on the Text Embeddings Inference (TEI) backend.
    The embedding of each Document is stored in the `embedding` field of the Document.

    You can use this component for embedding models hosted on Hugging Face Inference endpoints, the rate-limited
    Inference API tier:

    ```python
    from haystack.dataclasses import Document
    from haystack.components.embedders import HuggingFaceTEIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = HuggingFaceTEIDocumentEmbedder(
        model="BAAI/bge-small-en-v1.5", token="<your-token>"
    )

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    Or for embedding models hosted on paid https://huggingface.co/inference-endpoints endpoint, and/or your own custom
    TEI endpoint. In these two cases, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.dataclasses import Document
    from haystack.components.embedders import HuggingFaceTEIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = HuggingFaceTEIDocumentEmbedder(
        model="BAAI/bge-small-en-v1.5", url="<your-tei-endpoint-url>", token="<your-token>"
    )

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    Key Features and Compatibility:
        - **Primary Compatibility**: Designed to work seamlessly with any embedding model deployed using the TEI
        framework. For more information on TEI, visit https://github.com/huggingface/text-embeddings-inference.
        - **Hugging Face Inference Endpoints**: Supports inference of TEI embedding models deployed on Hugging Face
        Inference endpoints. For more details refer to https://huggingface.co/inference-endpoints.
        - **Inference API Support**: Supports inference of TEI embedding models hosted on the rate-limited Inference
        API tier. Learn more about the Inference API at: https://huggingface.co/inference-api
        Discover available embedding models using the following command:
        ```
        wget -qO- https://api-inference.huggingface.co/framework/sentence-transformers
        ```
        And simply use the model ID as the model parameter for this component. You'll also need to provide a valid
        Hugging Face API token as the token parameter.
        - **Custom TEI Endpoints**: Supports inference of embedding models deployed on custom TEI endpoints. Anyone can
        deploy their own TEI endpoint using the TEI framework. For more details refer
        to https://huggingface.co/inference-endpoints.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        url: Optional[str] = None,
        token: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a HuggingFaceTEIDocumentEmbedder component.

        :param model: A string representing the model id on HF Hub. Default is "BAAI/bge-small-en-v1.5".
        :param url: The URL of your self-deployed Text-Embeddings-Inference service or the URL of your paid HF Inference
                    Endpoint.
        :param token: The HuggingFace Hub token. This is needed if you are using a paid HF Inference Endpoint or serving
                      a private or gated model. It can be explicitly provided or automatically read from the environment
                      variable HF_API_TOKEN (recommended).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        transformers_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TEI endpoint URL provided: {url}")

        # The user does not need to provide a token if it is a local server or free public HF Inference Endpoint.
        token = token or os.environ.get("HF_API_TOKEN")

        check_valid_model(model, token)

        self.model = model
        self.url = url
        self.token = token
        self.client = InferenceClient(url or model, token=token)
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `token` value passed
        to the constructor.
        """
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # Don't send URL as it is sensitive information
        return {"model": self.model}

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            embeddings = self.client.feature_extraction(text=batch)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "HuggingFaceTEIDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the HuggingFaceTEITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
