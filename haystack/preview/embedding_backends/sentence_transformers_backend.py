from typing import List, Optional, Union, Dict
import hashlib
import numpy as np

from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install farm-haystack[inference]'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer


class _SentenceTransformersEmbeddingBackend:
    """
    Singleton class to manage SentenceTransformers embeddings.
    """

    _instances: Dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    def __new__(
        cls, model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        args_str = f"{model_name_or_path}{device}{use_auth_token}"
        instance_id = hashlib.md5(args_str.encode()).hexdigest()
        if instance_id in cls._instances:
            return cls._instances[instance_id]

        instance = super().__new__(cls)
        cls._instances[instance_id] = instance
        return instance

    def __init__(
        self, model_name_or_path: str, device: Optional[str] = None, use_auth_token: Union[bool, str, None] = None
    ):
        sentence_transformers_import.check()
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path, device=device, use_auth_token=use_auth_token
            )

    def embed(self, data: List[str], **kwargs) -> np.ndarray:
        embedding = self.model.encode(data, **kwargs)
        return embedding
