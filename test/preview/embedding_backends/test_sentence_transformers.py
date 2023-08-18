from unittest.mock import Mock, patch
import pytest
from haystack.preview.embedding_backends.sentence_transformers_backend import _SentenceTransformersEmbeddingBackend
import numpy as np


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_singleton_behavior(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackend(model_name_or_path="my_model", device="cpu")
    same_embedding_backend = _SentenceTransformersEmbeddingBackend("my_model", "cpu")
    another_embedding_backend = _SentenceTransformersEmbeddingBackend(model_name_or_path="another_model", device="cpu")

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_model_initialization(mock_sentence_transformer):
    _SentenceTransformersEmbeddingBackend(model_name_or_path="model", device="cpu")
    mock_sentence_transformer.assert_called_once_with(model_name_or_path="model", device="cpu", use_auth_token=None)


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_embedding_function_with_kwargs(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackend(model_name_or_path="model")
    fake_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    embedding_backend.model.encode.return_value = fake_embeddings

    data = ["sentence1", "sentence2"]
    result = embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)
    np.testing.assert_array_equal(result, fake_embeddings)
