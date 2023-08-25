from typing import List, Optional, Union

import numpy as np

from haystack.preview import component
from haystack.preview.embedding_backends.sentence_transformers_backend import (
    SentenceTransformersEmbeddingBackendFactory,
)


@component
class SentenceTransformersTextEmbedder:
    """
    A component for embedding strings using Sentence Transformers models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create a SentenceTransformersTextEmbedder component.

        :param model_name_or_path: Local path or name of the model in Hugging Face's model hub, such as ``'sentence-transformers/all-MiniLM-L6-v2'``.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                        If this parameter is set to `True`, then the token generated when running
                        `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        """

        self.model_name_or_path = model_name_or_path
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device
        self.use_auth_token = use_auth_token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model_name_or_path=self.model_name_or_path, device=self.device, use_auth_token=self.use_auth_token
            )

    @component.output_types(embeddings=List[np.ndarray])
    def run(self, texts: List[str]):
        """Embed a list of strings."""
        if not isinstance(texts, list) or not isinstance(texts[0], str):
            raise ValueError(
                "SentenceTransformersTextEmbedder expects a list of strings as input."
                "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
            )
        self.warm_up()
        texts_to_embed = [self.prefix + text + self.suffix for text in texts]
        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )
        return {"embeddings": embeddings}
