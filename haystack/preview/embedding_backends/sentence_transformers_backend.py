from typing import List, Optional, Union, Dict
import hashlib
import numpy as np

from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer


class SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: Dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(
        model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        args_string = f"{model_name_or_path}{device}{use_auth_token}"
        embedding_backend_id = hashlib.md5(args_string.encode()).hexdigest()

        if embedding_backend_id in SentenceTransformersEmbeddingBackendFactory._instances:
            return SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
        )
        SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersEmbeddingBackend:
    """
    Class to manage SentenceTransformers embeddings.
    """

    def __init__(
        self, model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        sentence_transformers_import.check()
        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
        )

    def embed(self, data: List[str], **kwargs) -> List[np.ndarray]:
        embedding = self.model.encode(data, **kwargs)
        return list(embedding)
