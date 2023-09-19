from typing import List, Optional, Union, Dict

from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install InstructorEmbedding'") as instructor_embeddings_import:
    from InstructorEmbedding import INSTRUCTOR


class _InstructorEmbeddingBackendFactory:
    """
    Factory class to create instances of INSTRUCTOR embedding backends.
    """

    _instances: Dict[str, "_InstructorEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(
        model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        embedding_backend_id = f"{model_name_or_path}{device}{use_auth_token}"

        if embedding_backend_id in _InstructorEmbeddingBackendFactory._instances:
            return _InstructorEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _InstructorEmbeddingBackend(
            model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
        )
        _InstructorEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _InstructorEmbeddingBackend:
    """
    Class to manage INSTRUCTOR embeddings.
    """

    def __init__(
        self, model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        instructor_embeddings_import.check()
        self.model = INSTRUCTOR(model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token)

    def embed(self, data: List[List[str]], **kwargs) -> List[List[float]]:
        embeddings = self.model.encode(data, **kwargs).tolist()
        return embeddings
