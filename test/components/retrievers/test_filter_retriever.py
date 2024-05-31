# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any, List

import pytest

from haystack import Pipeline, DeserializationError
from haystack.testing.factory import document_store_class
from haystack.components.retrievers.filter_retriever import FilterRetriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture()
def sample_docs():
    en_docs = [
        Document(content="Javascript is a popular programming language", meta={"lang": "en"}),
        Document(content="Python is a popular programming language", meta={"lang": "en"}),
        Document(content="A chromosome is a package of DNA ", meta={"lang": "en"}),
    ]
    de_docs = [
        Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
        Document(content="javascript ist eine beliebte Programmiersprache", meta={"lang": "de"}),
    ]
    all_docs = en_docs + de_docs
    return {"en_docs": en_docs, "de_docs": de_docs, "all_docs": all_docs}


@pytest.fixture()
def sample_document_store(sample_docs):
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(sample_docs["all_docs"])
    return doc_store


class TestFilterRetriever:
    @classmethod
    def _documents_equal(cls, docs1: List[Document], docs2: List[Document]) -> bool:
        # # Order doesn't matter; we sort before comparing
        docs1.sort(key=lambda x: x.id)
        docs2.sort(key=lambda x: x.id)
        return docs1 == docs2

    def test_init_default(self):
        retriever = FilterRetriever(InMemoryDocumentStore())
        assert retriever.filters is None

    def test_init_with_parameters(self):
        retriever = FilterRetriever(InMemoryDocumentStore(), filters={"lang": "en"})
        assert retriever.filters == {"lang": "en"}

    def test_to_dict(self):
        FilterDocStore = document_store_class("MyFakeStore", bases=(InMemoryDocumentStore,))
        document_store = FilterDocStore()
        document_store.to_dict = lambda: {"type": "FilterDocStore", "init_parameters": {}}
        component = FilterRetriever(document_store=document_store)

        data = component.to_dict()
        assert data == {
            "type": "haystack.components.retrievers.filter_retriever.FilterRetriever",
            "init_parameters": {"document_store": {"type": "FilterDocStore", "init_parameters": {}}, "filters": None},
        }

    def test_to_dict_with_custom_init_parameters(self):
        ds = InMemoryDocumentStore(index="test_to_dict_with_custom_init_parameters")
        serialized_ds = ds.to_dict()

        component = FilterRetriever(
            document_store=InMemoryDocumentStore(index="test_to_dict_with_custom_init_parameters"),
            filters={"lang": "en"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.retrievers.filter_retriever.FilterRetriever",
            "init_parameters": {"document_store": serialized_ds, "filters": {"lang": "en"}},
        }

    def test_from_dict(self):
        valid_data = {
            "type": "haystack.components.retrievers.filter_retriever.FilterRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "filters": {"lang": "en"},
            },
        }
        component = FilterRetriever.from_dict(valid_data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.filters == {"lang": "en"}

    def test_from_dict_without_docstore(self):
        data = {"type": "InMemoryBM25Retriever", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            FilterRetriever.from_dict(data)

    def test_retriever_init_filter(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "en"})
        result = retriever.run()

        assert "documents" in result
        assert len(result["documents"]) == 3
        assert TestFilterRetriever._documents_equal(result["documents"], sample_docs["en_docs"])

    def test_retriever_runtime_filter(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store)
        result = retriever.run(filters={"field": "lang", "operator": "==", "value": "en"})

        assert "documents" in result
        assert len(result["documents"]) == 3
        assert TestFilterRetriever._documents_equal(result["documents"], sample_docs["en_docs"])

    def test_retriever_init_filter_run_filter_override(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "en"})
        result = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert TestFilterRetriever._documents_equal(result["documents"], sample_docs["de_docs"])

    @pytest.mark.integration
    def test_run_with_pipeline(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "de"})

        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        result: Dict[str, Any] = pipeline.run(data={"retriever": {}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert TestFilterRetriever._documents_equal(results_docs, sample_docs["de_docs"])

        result: Dict[str, Any] = pipeline.run(
            data={"retriever": {"filters": {"field": "lang", "operator": "==", "value": "en"}}}
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert TestFilterRetriever._documents_equal(results_docs, sample_docs["en_docs"])
