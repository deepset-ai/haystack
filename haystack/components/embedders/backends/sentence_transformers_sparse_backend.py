# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport(message="Run 'pip install \"sentence-transformers>=5.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer, SparseEncoder


class _SentenceTransformersSparseEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: dict[str, "_SentenceTransformersSparseEncoderEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(  # pylint: disable=too-many-positional-arguments
        *,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
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


class _SentenceTransformersSparseEncoderEmbeddingBackend:
    """
    Class to manage Sparse embeddings from Sentence Transformers.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        *,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
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

    def embed(self, *, data: list[str], **kwargs) -> list[SparseEmbedding]:
        embeddings = self.model.encode(data, **kwargs).coalesce()  # type: ignore[attr-defined]

        # Extract the row indices, column indices, values, and batch size from the sparse tensor embeddings
        rows, columns = embeddings.indices()
        values = embeddings.values()
        batch_size = embeddings.size(0)

        sparse_embeddings: list[SparseEmbedding] = []
        for embedding in range(batch_size):
            # For each embedding in the batch, create a mask to select its corresponding indices and values
            mask = rows == embedding
            # Extract the column indices and values for the current embedding in the batch
            embedding_columns = columns[mask].tolist()
            embedding_values = values[mask].tolist()
            sparse_embeddings.append(SparseEmbedding(indices=embedding_columns, values=embedding_values))

        return sparse_embeddings
