# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from haystack.components.embedders.backends.sentence_transformers_sparse_backend import (
    _SentenceTransformersSparseEmbeddingBackendFactory,
)
from haystack.dataclasses.sparse_embedding import SparseEmbedding
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
    yet_another_embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu", trust_remote_code=True
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend
    assert yet_another_embedding_backend is not embedding_backend


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
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(model="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, attn_implementation="sdpa")

    embedding_backend.model.encode.assert_called_once_with(
        data, convert_to_tensor=False, convert_to_sparse_tensor=True, attn_implementation="sdpa"
    )


@patch("haystack.components.embedders.backends.sentence_transformers_sparse_backend.SparseEncoder")
def test_sparse_embedding_function(mock_sparse_encoder):
    """
    Test that the backend's embed method returns the correct sparse embeddings.
    """

    # Ensure the factory cache is cleared before each test.
    _SentenceTransformersSparseEmbeddingBackendFactory._instances = {}

    tensors = [
        torch.sparse_coo_tensor(torch.tensor([[1, 4]]), torch.tensor([0.5, 0.8])),
        torch.sparse_coo_tensor(torch.tensor([[2]]), torch.tensor([0.3])),
    ]
    mock_sparse_encoder.return_value.encode.return_value = tensors

    # Get the embedding backend
    embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(model="model")

    # Embed dummy data
    data = ["sentence1", "sentence2"]
    sparse_embeddings = embedding_backend.embed(data=data)

    # Expected output
    expected_embeddings = [
        SparseEmbedding(indices=[1, 4], values=[0.5, 0.8]),
        SparseEmbedding(indices=[2], values=[0.3]),
    ]

    assert len(sparse_embeddings) == len(expected_embeddings)
    for got, exp in zip(sparse_embeddings, expected_embeddings):
        assert got.indices == exp.indices
        assert got.values == pytest.approx(exp.values)
