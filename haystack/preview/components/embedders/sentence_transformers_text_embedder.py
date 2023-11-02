from typing import List, Optional, Union, Dict, Any

from haystack.preview import component, default_to_dict
from haystack.preview.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)


@component
class SentenceTransformersTextEmbedder:
    """
    A component for embedding strings using Sentence Transformers models.

    Usage example:
    ```python
    from haystack.preview.components.embedders import SentenceTransformersTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = SentenceTransformersTextEmbedder()
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        token: Union[bool, str, None] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create a SentenceTransformersTextEmbedder component.

        :param model_name_or_path: Local path or name of the model in Hugging Face's model hub,
            such as ``'sentence-transformers/all-mpnet-base-v2'``.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation.
            Defaults to CPU.
        :param token: The API token used to download private models from Hugging Face.
            If this parameter is set to `True`, then the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param prefix: A string to add to the beginning of each Document text before embedding.
            Can be used to prepend the text with an instruction, as required by some embedding models,
            such as E5 and bge.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        """

        self.model_name_or_path = model_name_or_path
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device or "cpu"
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name_or_path}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.token
            )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
            )
        if not hasattr(self, "embedding_backend"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = self.prefix + text + self.suffix
        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        return {"embedding": embedding}
