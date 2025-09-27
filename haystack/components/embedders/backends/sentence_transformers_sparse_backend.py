# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install \"sentence-transformers>=5.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SparseEncoder


class _SentenceTransformersSparseEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: dict[str, "_SentenceTransformersSparseEncoderEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(**kwargs):
        embedding_backend_id = json.dumps(kwargs, sort_keys=True, default=str)

        if embedding_backend_id in _SentenceTransformersSparseEmbeddingBackendFactory._instances:
            return _SentenceTransformersSparseEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersSparseEncoderEmbeddingBackend(**kwargs)

        _SentenceTransformersSparseEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersSparseEncoderEmbeddingBackend:
    """
    Class to manage Sparse embeddings from Sentence Transformers.
    """

    def __init__(self, **kwargs):
        sentence_transformers_import.check()

        auth_token = kwargs.get("auth_token")
        resolved_token = auth_token.resolve_value() if auth_token else None

        self.model = SparseEncoder(
            model_name_or_path=kwargs["model"],
            device=kwargs.get("device"),
            token=resolved_token,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            local_files_only=kwargs.get("local_files_only", False),
            model_kwargs=kwargs.get("model_kwargs"),
            tokenizer_kwargs=kwargs.get("tokenizer_kwargs"),
            config_kwargs=kwargs.get("config_kwargs"),
            backend=kwargs.get("backend", "torch"),
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
