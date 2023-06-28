from typing import Dict, Any, List

import pytest

from haystack.preview import Pipeline
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


class Test_MemoryRetriever(BaseTestComponent):
    @pytest.mark.unit
    def test_save_load(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(MemoryRetriever(document_store_name="memory"), tmp_path)

    @pytest.mark.unit
    def test_save_load_with_parameters(self, tmp_path):
        self.assert_can_be_saved_and_loaded_in_pipeline(
            MemoryRetriever(document_store_name="memory", top_k=5, scale_score=False), tmp_path
        )

    @pytest.mark.unit
    def test_init_default(self):
        retriever = MemoryRetriever(document_store_name="memory")
        assert retriever.document_store_name == "memory"
        assert retriever.defaults == {"filters": {}, "top_k": 10, "scale_score": True}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryRetriever(document_store_name="memory-test", top_k=5, scale_score=False)
        assert retriever.document_store_name == "memory-test"
        assert retriever.defaults == {"filters": {}, "top_k": 5, "scale_score": False}

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryRetriever(document_store_name="memory-test", top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_valid_run(self, mock_docs):
        top_k = 5
        ds = MemoryDocumentStore()
        ds.write_documents(mock_docs)
        mr = MemoryRetriever(document_store_name="memory", top_k=top_k)
        result: MemoryRetriever.Output = mr.run(data=MemoryRetriever.Input(query="PHP", stores={"memory": ds}))

        assert getattr(result, "documents")
        assert len(result.documents) == top_k
        assert result.documents[0].content == "PHP is a popular programming language"

    @pytest.mark.unit
    def test_invalid_run_wrong_store_name(self):
        # Test invalid run with wrong store name
        ds = MemoryDocumentStore()
        mr = MemoryRetriever(document_store_name="memory")
        with pytest.raises(ValueError, match=r"MemoryRetriever's document store 'memory' not found"):
            invalid_input_data = MemoryRetriever.Input(
                query="test", top_k=10, scale_score=True, stores={"invalid_store": ds}
            )
            mr.run(invalid_input_data)

    @pytest.mark.unit
    def test_invalid_run_wrong_store_type(self):
        # Test invalid run with wrong store type
        ds = MemoryDocumentStore()
        mr = MemoryRetriever(document_store_name="memory")
        with pytest.raises(ValueError, match=r"MemoryRetriever can only be used with a MemoryDocumentStore instance."):
            invalid_input_data = MemoryRetriever.Input(
                query="test", top_k=10, scale_score=True, stores={"memory": "not a MemoryDocumentStore"}
            )
            mr.run(invalid_input_data)

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
        mr = MemoryRetriever(document_store_name="memory")

        pipeline = Pipeline()
        pipeline.add_component("retriever", mr)
        pipeline.add_store("memory", ds)
        result: Dict[str, Any] = pipeline.run(data={"retriever": MemoryRetriever.Input(query=query)})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"].documents
        assert results_docs
        assert results_docs[0].content == query_result

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
        mr = MemoryRetriever(document_store_name="memory")

        pipeline = Pipeline()
        pipeline.add_component("retriever", mr)
        pipeline.add_store("memory", ds)
        result: Dict[str, Any] = pipeline.run(data={"retriever": MemoryRetriever.Input(query=query, top_k=top_k)})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"].documents
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].content == query_result
