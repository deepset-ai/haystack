from typing import List, Optional, Union, Dict

from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install \"sentence-transformers>=2.2.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer


class _SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: Dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(
        model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        embedding_backend_id = f"{model_name_or_path}{device}{use_auth_token}"

        if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
            return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]
        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
        )
        _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersEmbeddingBackend:
    """
    Class to manage Sentence Transformers embeddings.
    """

    def __init__(
        self, model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        sentence_transformers_import.check()
        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
        )

    def embed(self, data: List[str], **kwargs) -> List[List[float]]:
        embeddings = self.model.encode(data, **kwargs).tolist()
        return embeddings
