from typing import Dict, Any, List, Optional

import pytest

from canals.errors import ComponentDeserializationError

from haystack.preview import Pipeline
from haystack.preview.components.retrievers.memory import MemoryRetriever
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.components.base import BaseTestComponent

from haystack.preview.document_stores.protocols import DuplicatePolicy


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
    @pytest.mark.unit
    def test_save_load(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(), tmp_path)

    @pytest.mark.unit
    def test_save_load_with_parameters(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(top_k=5, scale_score=False), tmp_path)

    @pytest.mark.unit
    def test_to_dict(self):
        retriever = MemoryRetriever()
        data = retriever.to_dict()
        assert data == {
            "hash": id(retriever),
            "type": "MemoryRetriever",
            "document_store": None,
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        filters = {"content_type": ["text"]}
        top_k = 42
        scale_score = False
        retriever = MemoryRetriever(filters=filters, top_k=top_k, scale_score=scale_score)
        data = retriever.to_dict()
        assert data == {
            "hash": id(retriever),
            "type": "MemoryRetriever",
            "document_store": None,
            "init_parameters": {"filters": {"content_type": ["text"]}, "top_k": 42, "scale_score": False},
        }

    @pytest.mark.unit
    def test_to_dict_with_store_instance(self):
        retriever = MemoryRetriever()
        retriever.document_store = MemoryDocumentStore()
        data = retriever.to_dict()
        assert data == {
            "hash": id(retriever),
            "type": "MemoryRetriever",
            "document_store": {
                "hash": id(retriever.document_store),
                "type": "MemoryDocumentStore",
                "init_parameters": {
                    "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                    "bm25_algorithm": "BM25Okapi",
                    "bm25_parameters": {},
                },
            },
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "hash": 1234,
            "type": "MemoryRetriever",
            "document_store": None,
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }
        retriever = MemoryRetriever.from_dict(data)
        assert retriever._document_store == None
        assert retriever._document_store_name == ""
        assert retriever.filters == None
        assert retriever.top_k == 10
        assert retriever.scale_score

    @pytest.mark.unit
    def test_from_dict_with_init_parameters(self):
        data = {
            "hash": 1234,
            "type": "MemoryRetriever",
            "document_store": None,
            "init_parameters": {"filters": {"content_type": ["text"]}, "top_k": 42, "scale_score": False},
        }
        retriever = MemoryRetriever.from_dict(data)
        assert retriever._document_store == None
        assert retriever._document_store_name == ""
        assert retriever.filters == {"content_type": ["text"]}
        assert retriever.top_k == 42
        assert not retriever.scale_score

    @pytest.mark.unit
    def test_from_dict_with_store_instance(self):
        data = {
            "hash": 1234,
            "type": "MemoryRetriever",
            "document_store": {
                "hash": 5678,
                "type": "MemoryDocumentStore",
                "init_parameters": {
                    "bm25_tokenization_regex": r"(?u)\b\w\w+\b",
                    "bm25_algorithm": "BM25Okapi",
                    "bm25_parameters": {},
                },
            },
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }
        retriever = MemoryRetriever.from_dict(data)
        assert isinstance(retriever._document_store, MemoryDocumentStore)
        assert retriever._document_store_name == ""
        assert retriever.filters == None
        assert retriever.top_k == 10
        assert retriever.scale_score

    @pytest.mark.unit
    def test_from_dict_without_type(self):
        data = {
            "hash": 1234,
            "document_store": None,
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }
        with pytest.raises(ComponentDeserializationError, match="Missing 'type' in component serialization data"):
            MemoryRetriever.from_dict(data)

    def test_from_dict_with_wrong_type(self, request):
        # We use the test function name as component type to make sure it's not registered.
        # Since the registry is global we risk to have a component with the same type registered in another test.
        component_type = request.node.name
        data = {
            "hash": 1234,
            "type": component_type,
            "document_store": None,
            "init_parameters": {"filters": None, "top_k": 10, "scale_score": True},
        }
        with pytest.raises(
            ComponentDeserializationError,
            match=f"Component '{component_type}' can't be deserialized as 'MemoryRetriever'",
        ):
            MemoryRetriever.from_dict(data)

    @pytest.mark.unit
    def test_init_default(self):
        retriever = MemoryRetriever()
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.scale_score

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryRetriever(filters={"name": "test.txt"}, top_k=5, scale_score=False)
        assert retriever.filters == {"name": "test.txt"}
        assert retriever.top_k == 5
        assert not retriever.scale_score

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryRetriever(top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_valid_run(self, mock_docs):
        top_k = 5
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)

        retriever = MemoryRetriever(top_k=top_k)
        retriever.document_store = ds
        result = retriever.run(queries=["PHP", "Java"])

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert len(result["documents"][0]) == top_k
        assert len(result["documents"][1]) == top_k
        assert result["documents"][0][0].content == "PHP is a popular programming language"
        assert result["documents"][1][0].content == "Java is a popular programming language"

    @pytest.mark.unit
    def test_invalid_run_no_store(self):
        retriever = MemoryRetriever()
        with pytest.raises(
            ValueError,
            match="MemoryRetriever needs a DocumentStore to run: set the DocumentStore instance to the self.document_store attribute",
        ):
            retriever.run(queries=["test"])

    @pytest.mark.unit
    def test_invalid_run_not_a_store(self):
        class MockStore:
            ...

        retriever = MemoryRetriever()
        with pytest.raises(ValueError, match="'MockStore' is not decorate with @document_store."):
            retriever.document_store = MockStore()

    @pytest.mark.unit
    def test_invalid_run_wrong_store_type(self):
        class MockStore:
            def count_documents(self) -> int:
                return 0

            def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
                return []

            def write_documents(
                self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
            ) -> None:
                return None

            def delete_documents(self, document_ids: List[str]) -> None:
                return None

        retriever = MemoryRetriever()
        with pytest.raises(ValueError, match="'MockStore' is not decorate with @document_store."):
            retriever.document_store = MockStore()

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
        retriever = MemoryRetriever()

        pipeline = Pipeline()
        pipeline.add_document_store("memory", ds)
        pipeline.add_component("retriever", retriever, document_store="memory")
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
        retriever = MemoryRetriever()

        pipeline = Pipeline()
        pipeline.add_document_store("memory", ds)
        pipeline.add_component("retriever", retriever, document_store="memory")
        result: Dict[str, Any] = pipeline.run(data={"retriever": {"queries": [query], "top_k": top_k}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert len(results_docs[0]) == top_k
        assert results_docs[0][0].content == query_result
