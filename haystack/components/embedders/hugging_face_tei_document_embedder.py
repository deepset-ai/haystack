import warnings
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model
from haystack.utils.url_validation import is_valid_url

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.22.0\"'") as huggingface_hub_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


# TODO: remove the default model in Haystack 2.3.0, as explained in the deprecation warning
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@component
class HuggingFaceTEIDocumentEmbedder:
    """
    A component for computing Document embeddings using HuggingFace Text-Embeddings-Inference endpoints.

    This component can be used with embedding models hosted on Hugging Face Inference endpoints,
    on the rate-limited Hugging Face Inference API or on your own custom TEI endpoint.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack.components.embedders import HuggingFaceTEIDocumentEmbedder
    from haystack.utils import Secret

    doc = Document(content="I love pizza!")

    document_embedder = HuggingFaceTEIDocumentEmbedder(
        model="BAAI/bge-small-en-v1.5", token=Secret.from_token("<your-api-key>")
    )

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        url: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a HuggingFaceTEIDocumentEmbedder component.

        :param model:
            An optional string representing the ID of the model on HuggingFace Hub.
            If not provided, the `url` parameter must be set to a valid TEI endpoint.
            In case both `model` and `url` are provided, the `url` parameter will be used.
        :param url:
            An optional string representing the URL of your self-deployed Text-Embeddings-Inference service
            or the URL of your paid HF Inference Endpoint.
            If not provided, the `model` parameter must be set to a valid model ID and the Hugging Face Inference API
            will be used.
            In case both `model` and `url` are provided, the `url` parameter will be used.
        :param token:
            The HuggingFace Hub token. This is needed if you are using a paid HF Inference Endpoint or serving
            a private or gated model.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            If True shows a progress bar when running.
        :param meta_fields_to_embed:
            List of meta fields that will be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        """
        huggingface_hub_import.check()

        if not model and not url:
            warnings.warn(
                f"Neither `model` nor `url` is provided. The component will use the default model: {DEFAULT_MODEL}. "
                "This behavior is deprecated and will be removed in Haystack 2.3.0.",
                DeprecationWarning,
            )
            model = DEFAULT_MODEL
        elif model and url:
            logger.warning("Both `model` and `url` are provided. The `model` parameter will be ignored. ")

        if url and not is_valid_url(url):
            raise ValueError(f"Invalid TEI endpoint URL provided: {url}")
        if not url and model:
            check_valid_model(model, HFModelType.EMBEDDING, token)

        self.model = model
        self.url = url
        self.token = token
        self.client = InferenceClient(url or model, token=token.resolve_value() if token else None)
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
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
            token=self.token.to_dict() if self.token else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTEIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

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

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
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
