from unittest.mock import patch
import pytest
from haystack.preview.embedding_backends.instructor_backend import _InstructorEmbeddingBackendFactory


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.instructor_backend.INSTRUCTOR")
def test_factory_behavior(mock_instructor):
    embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-xl", device="cpu"
    )
    same_embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend("hkunlp/instructor-xl", "cpu")
    another_embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base", device="cpu"
    )

    assert same_embedding_backend is embedding_backend
    assert another_embedding_backend is not embedding_backend


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.instructor_backend.INSTRUCTOR")
def test_model_initialization(mock_instructor):
    _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token="huggingface_auth_token"
    )
    mock_instructor.assert_called_once_with(
        model_name_or_path="hkunlp/instructor-base", device="cpu", use_auth_token="huggingface_auth_token"
    )


@pytest.mark.unit
@patch("haystack.preview.embedding_backends.instructor_backend.INSTRUCTOR")
def test_embedding_function_with_kwargs(mock_instructor):
    embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
        model_name_or_path="hkunlp/instructor-base"
    )

    data = [["instruction", "sentence1"], ["instruction", "sentence2"]]
    embedding_backend.embed(data=data, normalize_embeddings=True)

    embedding_backend.model.encode.assert_called_once_with(data, normalize_embeddings=True)
