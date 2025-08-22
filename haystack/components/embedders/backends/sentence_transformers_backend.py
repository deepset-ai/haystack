# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Union

from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport(message="Run 'pip install \"sentence-transformers>=5.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer, SparseEncoder

with LazyImport(message="Run 'pip install \"pillow\"'") as pillow_import:
    from PIL.Image import Image


class _SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(  # pylint: disable=too-many-positional-arguments
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        embedding_backend_id = f"{model}{device}{auth_token}{truncate_dim}{backend}"

        if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
            return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

        _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersSparseEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: Dict[str, "_SentenceTransformersSparseEncoderEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(  # pylint: disable=too-many-positional-arguments
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        embedding_backend_id = f"{model}{device}{auth_token}{backend}"

        if embedding_backend_id in _SentenceTransformersSparseEmbeddingBackendFactory._instances:
            return _SentenceTransformersSparseEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersSparseEncoderEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

        _SentenceTransformersSparseEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
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
        local_files_only: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        sentence_transformers_import.check()

        self.model = SentenceTransformer(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(self, data: Union[list[str], list["Image"]], **kwargs: Any) -> list[list[float]]:
        # Sentence Transformers encode can work with Images, but the type hint does not reflect that
        # https://sbert.net/examples/sentence_transformer/applications/image-search
        embeddings = self.model.encode(data, **kwargs).tolist()  # type: ignore[arg-type]
        return embeddings


class _SentenceTransformersSparseEncoderEmbeddingBackend:
    """
    Class to manage Sparse embeddings from Sentence Transformers.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        sentence_transformers_import.check()

        self.model = SparseEncoder(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(self, data: List[str], **kwargs) -> List[SparseEmbedding]:
        embeddings = self.model.encode(data, **kwargs).coalesce()

        rows, columns = embeddings.indices()
        values = embeddings.values()
        batch_size = embeddings.size(0)

        sparse_embeddings: List[SparseEmbedding] = []
        for embedding in range(batch_size):
            mask = rows == embedding
            embedding_columns = columns[mask].tolist()
            embedding_values = values[mask].tolist()
            sparse_embeddings.append(SparseEmbedding(indices=embedding_columns, values=embedding_values))

        return sparse_embeddings
