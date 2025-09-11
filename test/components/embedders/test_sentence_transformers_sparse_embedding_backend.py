# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from haystack.components.embedders.backends.sentence_transformers_sparse_backend import (
    _SentenceTransformersSparseEmbeddingBackendFactory,
)
from haystack.utils.auth import Secret


@patch("haystack.components.embedders.backends.sentence_transformers_sparse_backend.SparseEncoder")
def test_sparse_factory_behavior(mock_sparse_encoder):
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    another_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="another_model", device="cpu"
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend


@patch("haystack.components.embedders.backends.sentence_transformers_sparse_backend.SparseEncoder")
def test_sparse_model_initialization(mock_sparse_encoder):
    _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="model",
        device="cpu",
        auth_token=Secret.from_token("fake-api-token"),
        trust_remote_code=True,
        local_files_only=True,
        backend="torch",
    )
    mock_sparse_encoder.assert_called_once_with(
        model_name_or_path="model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs=None,
        tokenizer_kwargs=None,
        config_kwargs=None,
        backend="torch",
    )


@patch("haystack.components.embedders.backends.sentence_transformers_sparse_backend.SparseEncoder")
def test_sparse_embedding_function_with_kwargs(mock_sparse_encoder):
    indices = torch.tensor([[0, 1], [1, 3]])
    values = torch.tensor([0.5, 0.7])
    mock_sparse_encoder.return_value.encode.return_value = torch.sparse_coo_tensor(indices, values, (2, 5))

    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(model="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, attn_implementation="sdpa")

    embedding_backend.model.encode.assert_called_once_with(data, attn_implementation="sdpa")
