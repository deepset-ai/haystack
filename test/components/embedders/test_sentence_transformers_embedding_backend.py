# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np

from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.utils.auth import Secret


@patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer")
def test_factory_behavior(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu"
    )
    another_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="another_model", device="cpu"
    )
    yet_another_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="my_model", device="cpu", trust_remote_code=True
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend
    assert yet_another_embedding_backend is not embedding_backend


@patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer")
def test_model_initialization(mock_sentence_transformer):
    _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model="model",
        device="cpu",
        auth_token=Secret.from_token("fake-api-token"),
        trust_remote_code=True,
        local_files_only=True,
        truncate_dim=256,
        backend="torch",
    )
    mock_sentence_transformer.assert_called_once_with(
        model_name_or_path="model",
        device="cpu",
        token="fake-api-token",
        trust_remote_code=True,
        revision=None,
        local_files_only=True,
        truncate_dim=256,
        model_kwargs=None,
        tokenizer_kwargs=None,
        config_kwargs=None,
        backend="torch",
    )


@patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer")
def test_embedding_function_with_kwargs(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)


@patch("haystack.components.embedders.backends.sentence_transformers_backend.quantize_embeddings")
@patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer")
def test_embedding_function_with_quantization_ranges(mock_sentence_transformer, mock_quantize_embeddings):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="quantized_model")
    embedding_backend.model.encode.return_value = np.array([[0.1, 0.2]])
    mock_quantize_embeddings.return_value = np.array([[12, 34]], dtype=np.int8)

    data = ["sentence"]
    ranges = [[-1.0, -1.0], [1.0, 1.0]]
    result = embedding_backend.embed(data=data, precision="int8", quantization_ranges=ranges)

    embedding_backend.model.encode.assert_called_once_with(data, precision="float32")
    assert mock_quantize_embeddings.call_count == 1
    _, called_kwargs = mock_quantize_embeddings.call_args
    assert called_kwargs["precision"] == "int8"
    assert np.array_equal(called_kwargs["ranges"], np.asarray(ranges))
    assert result == [[12, 34]]


@patch("haystack.components.embedders.backends.sentence_transformers_backend.quantize_embeddings")
@patch("haystack.components.embedders.backends.sentence_transformers_backend.SentenceTransformer")
def test_embedding_function_without_quantization_ranges(mock_sentence_transformer, mock_quantize_embeddings):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model="quantized_model_2")

    data = ["sentence"]
    embedding_backend.embed(data=data, precision="int8", quantization_ranges=None)

    embedding_backend.model.encode.assert_called_once_with(data, precision="int8")
    mock_quantize_embeddings.assert_not_called()
