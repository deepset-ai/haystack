# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import Document, component
from haystack.components.retrievers.filter_retriever import FilterRetriever
from haystack.components.retrievers.multi_filter_retriever import MultiFilterRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(content="English text", id="doc1", meta={"lang": "en"}),
        Document(content="German text", id="doc2", meta={"lang": "de"}),
    ]


@pytest.fixture
def sample_document_store(sample_documents: list[Document]) -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP).run(documents=sample_documents)
    return document_store


@pytest.fixture
def sample_filters() -> list[dict[str, Any]]:
    return [
        {"field": "meta.lang", "operator": "==", "value": "en"},
        {"field": "meta.lang", "operator": "==", "value": "de"},
    ]


class TestMultiFilterRetriever:
    def test_init_default(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi = MultiFilterRetriever(retriever=retriever)

        assert multi.retriever == retriever
        assert multi.max_workers == 3

    def test_init_with_parameters(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi = MultiFilterRetriever(retriever=retriever, max_workers=2)

        assert multi.max_workers == 2

    def test_run_empty_filters(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        multi = MultiFilterRetriever(retriever=FilterRetriever(document_store=in_memory_doc_store))

        result = multi.run(filters=[])

        assert result == {"documents": []}

    def test_run_multiple_filters(
        self, sample_document_store: InMemoryDocumentStore, sample_filters: list[dict[str, Any]]
    ) -> None:
        multi = MultiFilterRetriever(retriever=FilterRetriever(document_store=sample_document_store))

        result = multi.run(filters=sample_filters)

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert {doc.meta["lang"] for doc in result["documents"]} == {"en", "de"}

    def test_run_single_filter(self, sample_document_store: InMemoryDocumentStore) -> None:
        multi = MultiFilterRetriever(retriever=FilterRetriever(document_store=sample_document_store))

        result = multi.run(filters=[{"field": "meta.lang", "operator": "==", "value": "en"}])

        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["lang"] == "en"

    def test_deduplication(self) -> None:
        doc1 = Document(content="A", id="doc1", score=0.9)
        doc2 = Document(content="B", id="doc2", score=0.8)
        doc3 = Document(content="A", id="doc1", score=0.7)

        @component
        class MockRetriever:
            @component.output_types(documents=list[Document])
            def run(self, filters: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, list[Document]]:
                return {"documents": [doc1, doc2, doc3]}

        multi = MultiFilterRetriever(retriever=MockRetriever(), max_workers=1)

        result = multi.run(filters=[{}, {}])

        assert len(result["documents"]) == 2
        assert {doc.id for doc in result["documents"]} == {"doc1", "doc2"}

    def test_to_dict(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi = MultiFilterRetriever(retriever=retriever, max_workers=2)

        data = component_to_dict(multi, "multi_filter")

        assert data["type"] == "haystack.components.retrievers.multi_filter_retriever.MultiFilterRetriever"
        assert data["init_parameters"]["max_workers"] == 2

    def test_from_dict(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi = MultiFilterRetriever(retriever=retriever, max_workers=2)

        serialized = component_to_dict(multi, "multi_filter")
        deserialized = component_from_dict(MultiFilterRetriever, serialized, "multi_filter")

        assert isinstance(deserialized, MultiFilterRetriever)
        assert deserialized.max_workers == 2
