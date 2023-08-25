from unittest.mock import patch
import pytest
from haystack.preview.embedding_backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_factory_behavior(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="my_model", device="cpu"
    )
    same_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend("my_model", "cpu")
    another_embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="another_model", device="cpu"
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_model_initialization(mock_sentence_transformer):
    _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="model", device="cpu", use_auth_token="my_token"
    )
    mock_sentence_transformer.assert_called_once_with(
        model_name_or_path="model", device="cpu", use_auth_token="my_token"
    )


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.sentence_transformers_backend.SentenceTransformer")
def test_embedding_function_with_kwargs(mock_sentence_transformer):
    embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(model_name_or_path="model")

    data = ["sentence1", "sentence2"]
    embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)
