from unittest.mock import patch, Mock
import pytest

from haystack.dataclasses import Answer
from haystack.testing.factory import document_store_class
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipeline_utils.rag import build_rag_pipeline


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.ChatCompletion.create", autospec=True) as mock_chat_completion_create:
        # mimic the response from the OpenAI API
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"

        mock_message = Mock()
        mock_message.content = "I'm fine, thanks. How are you?"
        mock_message.role = "user"

        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock()
        mock_response.usage.items.return_value = [
            ("prompt_tokens", 57),
            ("completion_tokens", 40),
            ("total_tokens", 97),
        ]
        mock_response.choices = [mock_choice]
        mock_chat_completion_create.return_value = mock_response
        yield mock_chat_completion_create


def test_rag_pipeline(mock_chat_completion):
    rag_pipe = build_rag_pipeline(document_store=InMemoryDocumentStore())
    answer = rag_pipe.run(query="question")
    assert isinstance(answer, Answer)


def test_rag_pipeline_other_docstore():
    FakeStore = document_store_class("FakeStore")
    with pytest.raises(ValueError, match="InMemoryDocumentStore"):
        assert build_rag_pipeline(document_store=FakeStore())


def test_rag_pipeline_embedder_exist_if_model_is_given():
    rag_pipe = build_rag_pipeline(
        document_store=InMemoryDocumentStore(), embedding_model="sentence-transformers/all-mpnet-base-v2"
    )
    assert "text_embedder" in rag_pipe.pipeline.graph.nodes
