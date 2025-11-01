# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Literal, Optional

from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport(message="Run 'pip install \"sentence-transformers>=5.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SparseEncoder


class _SentenceTransformersSparseEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: dict[str, "_SentenceTransformersSparseEncoderEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(
        *,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        cache_params = {
            "model": model,
            "device": device,
            "auth_token": auth_token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
            "config_kwargs": config_kwargs,
            "backend": backend,
        }

        embedding_backend_id = json.dumps(cache_params, sort_keys=True, default=str)

        if embedding_backend_id in _SentenceTransformersSparseEmbeddingBackendFactory._instances:
            return _SentenceTransformersSparseEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersSparseEncoderEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            revision=revision,
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

    def __init__(
        self,
        *,
        model: str,
        device: Optional[str] = None,
        auth_token: Optional[Secret] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
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
            revision=revision,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(self, *, data: list[str], **kwargs) -> list[SparseEmbedding]:
        embeddings_list = self.model.encode(
            data,
            convert_to_tensor=False,  # output is a list of individual tensors
            convert_to_sparse_tensor=True,
            **kwargs,
        )

        sparse_embeddings: list[SparseEmbedding] = []
        for embedding_tensor in embeddings_list:
            # encode returns a list of tensors with the parameters above, but the type hint is too broad
            embedding_tensor = embedding_tensor.coalesce()  # type: ignore[union-attr]
            indices = embedding_tensor.indices()[0].tolist()  # Only column indices
            values = embedding_tensor.values().tolist()
            sparse_embeddings.append(SparseEmbedding(indices=indices, values=values))

        return sparse_embeddings
