from typing import Dict, Any

import pytest

from haystack.preview import Pipeline, DeserializationError
from haystack.preview.testing.factory import document_store_class
from haystack.preview.components.retrievers.memory_bm25_retriever import MemoryBM25Retriever
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore


@pytest.fixture()
def mock_docs():
    return [
        Document(text="Javascript is a popular programming language"),
        Document(text="Java is a popular programming language"),
        Document(text="Python is a popular programming language"),
        Document(text="Ruby is a popular programming language"),
        Document(text="PHP is a popular programming language"),
    ]


class TestMemoryBM25Retriever:
    @pytest.mark.unit
    def test_init_default(self):
        retriever = MemoryBM25Retriever(MemoryDocumentStore())
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.scale_score

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryBM25Retriever(MemoryDocumentStore(), filters={"name": "test.txt"}, top_k=5, scale_score=False)
        assert retriever.filters == {"name": "test.txt"}
        assert retriever.top_k == 5
        assert not retriever.scale_score

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryBM25Retriever(MemoryDocumentStore(), top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_to_dict(self):
        MyFakeStore = document_store_class("MyFakeStore", bases=(MemoryDocumentStore,))
        document_store = MyFakeStore()
        document_store.to_dict = lambda: {"type": "MyFakeStore", "init_parameters": {}}
        component = MemoryBM25Retriever(document_store=document_store)

        data = component.to_dict()
        assert data == {
            "type": "MemoryBM25Retriever",
            "init_parameters": {
                "document_store": {"type": "MyFakeStore", "init_parameters": {}},
                "filters": None,
                "top_k": 10,
                "scale_score": True,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        MyFakeStore = document_store_class("MyFakeStore", bases=(MemoryDocumentStore,))
        document_store = MyFakeStore()
        document_store.to_dict = lambda: {"type": "MyFakeStore", "init_parameters": {}}
        component = MemoryBM25Retriever(
            document_store=document_store, filters={"name": "test.txt"}, top_k=5, scale_score=False
        )
        data = component.to_dict()
        assert data == {
            "type": "MemoryBM25Retriever",
            "init_parameters": {
                "document_store": {"type": "MyFakeStore", "init_parameters": {}},
                "filters": {"name": "test.txt"},
                "top_k": 5,
                "scale_score": False,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        document_store_class("MyFakeStore", bases=(MemoryDocumentStore,))
        data = {
            "type": "MemoryBM25Retriever",
            "init_parameters": {
                "document_store": {"type": "MyFakeStore", "init_parameters": {}},
                "filters": {"name": "test.txt"},
                "top_k": 5,
            },
        }
        component = MemoryBM25Retriever.from_dict(data)
        assert isinstance(component.document_store, MemoryDocumentStore)
        assert component.filters == {"name": "test.txt"}
        assert component.top_k == 5
        assert component.scale_score

    @pytest.mark.unit
    def test_from_dict_without_docstore(self):
        data = {"type": "MemoryBM25Retriever", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            MemoryBM25Retriever.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_without_docstore_type(self):
        data = {"type": "MemoryBM25Retriever", "init_parameters": {"document_store": {"init_parameters": {}}}}
        with pytest.raises(DeserializationError, match="Missing 'type' in document store's serialization data"):
            MemoryBM25Retriever.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "MemoryBM25Retriever",
            "init_parameters": {"document_store": {"type": "NonexistingDocstore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="DocumentStore type 'NonexistingDocstore' not found"):
            MemoryBM25Retriever.from_dict(data)

    @pytest.mark.unit
    def test_retriever_valid_run(self, mock_docs):
        top_k = 5
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)

        retriever = MemoryBM25Retriever(ds, top_k=top_k)
        result = retriever.run(query="PHP")

        assert "documents" in result
        assert len(result["documents"]) == top_k
        assert result["documents"][0].text == "PHP is a popular programming language"

    @pytest.mark.unit
    def test_invalid_run_wrong_store_type(self):
        SomeOtherDocumentStore = document_store_class("SomeOtherDocumentStore")
        with pytest.raises(ValueError, match="document_store must be an instance of MemoryDocumentStore"):
            MemoryBM25Retriever(SomeOtherDocumentStore())

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
        retriever = MemoryBM25Retriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = pipeline.run(data={"retriever": {"query": query}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert results_docs[0].text == query_result

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
        retriever = MemoryBM25Retriever(ds)

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = pipeline.run(data={"retriever": {"query": query, "top_k": top_k}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].text == query_result
