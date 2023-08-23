from typing import Dict, Any

import pytest

from haystack.preview import Pipeline
from haystack.preview.testing.factory import document_store_class
from haystack.preview.components.retrievers.memory import MemoryRetriever
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.components.base import BaseTestComponent


@pytest.fixture()
def mock_docs():
    return [
        Document.from_dict({"content": "Javascript is a popular programming language"}),
        Document.from_dict({"content": "Java is a popular programming language"}),
        Document.from_dict({"content": "Python is a popular programming language"}),
        Document.from_dict({"content": "Ruby is a popular programming language"}),
        Document.from_dict({"content": "PHP is a popular programming language"}),
    ]


class TestMemoryRetriever(BaseTestComponent):
    # TODO: We're going to rework these tests when we'll remove BaseTestComponent.
    # We also need to implement `to_dict` and `from_dict` to test this properly.
    # @pytest.mark.unit
    # def test_save_load(self, tmp_path):
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(MemoryDocumentStore()), tmp_path)

    # @pytest.mark.unit
    # def test_save_load_with_parameters(self, tmp_path):
    #     self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(top_k=5, scale_score=False), tmp_path)

    @pytest.mark.unit
    def test_init_default(self):
        retriever = MemoryRetriever(MemoryDocumentStore())
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.scale_score

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryRetriever(MemoryDocumentStore(), filters={"name": "test.txt"}, top_k=5, scale_score=False)
        assert retriever.filters == {"name": "test.txt"}
        assert retriever.top_k == 5
        assert not retriever.scale_score

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryRetriever(MemoryDocumentStore(), top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_valid_run(self, mock_docs):
        top_k = 5
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)

        retriever = MemoryRetriever(ds, top_k=top_k)
        result = retriever.run(queries=["PHP", "Java"])

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert len(result["documents"][0]) == top_k
        assert len(result["documents"][1]) == top_k
        assert result["documents"][0][0].content == "PHP is a popular programming language"
        assert result["documents"][1][0].content == "Java is a popular programming language"

    @pytest.mark.unit
    def test_invalid_run_wrong_store_type(self):
        SomeOtherDocumentStore = document_store_class("SomeOtherDocumentStore")
        with pytest.raises(ValueError, match="document_store must be an instance of MemoryDocumentStore"):
            MemoryRetriever(SomeOtherDocumentStore())

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result",
        [
            ("Javascript", "Javascript is a popular programming language"),
            ("Java", "Java is a popular programming language"),
        ],
    )
    def test_run_with_pipeline(self, mock_docs, query: str, query_result: str):
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)
        retriever = MemoryRetriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = pipeline.run(data={"retriever": {"queries": [query]}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert results_docs[0][0].content == query_result

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "query, query_result, top_k",
        [
            ("Javascript", "Javascript is a popular programming language", 1),
            ("Java", "Java is a popular programming language", 2),
            ("Ruby", "Ruby is a popular programming language", 3),
        ],
    )
    def test_run_with_pipeline_and_top_k(self, mock_docs, query: str, query_result: str, top_k: int):
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)
        retriever = MemoryRetriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = pipeline.run(data={"retriever": {"queries": [query], "top_k": top_k}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs[0]) == top_k
        assert results_docs[0][0].content == query_result
