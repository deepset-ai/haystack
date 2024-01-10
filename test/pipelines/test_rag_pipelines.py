import os

import pytest

from haystack.dataclasses import Answer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline_utils.rag import build_rag_pipeline
from haystack.testing.factory import document_store_class


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
@pytest.mark.integration
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
