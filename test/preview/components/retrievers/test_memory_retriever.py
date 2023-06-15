from typing import Dict, Any

import pytest

from haystack.preview import Pipeline
from haystack.preview.components.retrievers.memory import MemoryRetriever
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.components.base import BaseTestComponent


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
        assert retriever.defaults["top_k"] == 10
        assert retriever.defaults["scale_score"]

    @pytest.mark.unit
    def test_init_with_parameters(self):
        retriever = MemoryRetriever(document_store_name="memory-test", top_k=5, scale_score=False)
        assert retriever.document_store_name == "memory-test"
        assert retriever.defaults["top_k"] == 5
        assert not retriever.defaults["scale_score"]

    @pytest.mark.unit
    def test_init_with_invalid_top_k_parameter(self):
        with pytest.raises(ValueError, match="top_k must be > 0, but got -2"):
            MemoryRetriever(document_store_name="memory-test", top_k=-2, scale_score=False)

    @pytest.mark.unit
    def test_run(self):
        docs = [
            Document.from_dict({"content": "Javascript is a popular programming language"}),
            Document.from_dict({"content": "Java is a popular programming language"}),
            Document.from_dict({"content": "Python is a popular programming language"}),
            Document.from_dict({"content": "Ruby is a popular programming language"}),
            Document.from_dict({"content": "PHP is a popular programming language"}),
        ]
        ds = MemoryDocumentStore()
        ds.write_documents(docs)
        mr = MemoryRetriever(document_store_name="memory")
        result: MemoryRetriever.Output = mr.run(data=MemoryRetriever.Input(query="PHP", stores={"memory": ds}))

        assert result.documents
        assert len(result.documents) == len(docs)
        assert result.documents[0].content == "PHP is a popular programming language"

    @pytest.mark.integration
    def test_run_with_pipeline(self):
        top_k = 1
        docs = [
            Document.from_dict({"content": "Javascript is a popular programming language"}),
            Document.from_dict({"content": "Java is a popular programming language"}),
            Document.from_dict({"content": "Python is a popular programming language"}),
            Document.from_dict({"content": "Ruby is a popular programming language"}),
            Document.from_dict({"content": "PHP is a popular programming language"}),
        ]
        ds = MemoryDocumentStore()
        ds.write_documents(docs)
        mr = MemoryRetriever(document_store_name="memory")

        pipeline = Pipeline()
        pipeline.add_component("retriever", mr)
        pipeline.add_store("memory", ds)
        result: Dict[str, Any] = pipeline.run(data={"retriever": MemoryRetriever.Input(query="Java", top_k=top_k)})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"].documents
        assert results_docs
        assert len(results_docs) == top_k
        assert results_docs[0].content == "Java is a popular programming language"
