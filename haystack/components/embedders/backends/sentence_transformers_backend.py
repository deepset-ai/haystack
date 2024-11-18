# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport(message="Run 'pip install \"sentence-transformers>=3.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer


class _SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: Dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(  # pylint: disable=too-many-positional-arguments
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        embedding_backend_id = f"{model}{device}{auth_token}{truncate_dim}"

        if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
            return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]
        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
        )
        _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersEmbeddingBackend:
    """
    Class to manage Sentence Transformers embeddings.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        sentence_transformers_import.check()
        self.model = SentenceTransformer(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
        )

    def embed(self, data: List[str], **kwargs) -> List[List[float]]:
        embeddings = self.model.encode(data, **kwargs).tolist()
        return embeddings
