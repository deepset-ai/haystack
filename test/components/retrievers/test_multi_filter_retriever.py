# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import Document, component
from haystack.components.retrievers import FilterRetriever, MultiFilterRetriever
from haystack.components.writers import DocumentWriter
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


class TestMultiFilterRetriever:
    def test_init_with_default_parameters(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        filter_retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi_retriever = MultiFilterRetriever(retriever=filter_retriever)

        assert multi_retriever.retriever == filter_retriever
        assert multi_retriever.max_workers == 3

    def test_init_with_custom_parameters(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        filter_retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi_retriever = MultiFilterRetriever(retriever=filter_retriever, max_workers=2)

        assert multi_retriever.retriever == filter_retriever
        assert multi_retriever.max_workers == 2

    def test_run_with_empty_filters(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        multi_retriever = MultiFilterRetriever(retriever=FilterRetriever(document_store=in_memory_doc_store))
        result = multi_retriever.run(filters=[])

        assert result == {"documents": []}

    @pytest.mark.parametrize(
        ("filters", "expected_languages"),
        [
            (
                [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.lang", "operator": "==", "value": "de"},
                ],
                {"en", "de"},
            )
        ],
    )
    def test_run_with_multiple_filters(
        self, sample_document_store: InMemoryDocumentStore, filters: list[dict[str, Any]], expected_languages: set[str]
    ) -> None:
        filter_retriever = FilterRetriever(document_store=sample_document_store)
        multi_retriever = MultiFilterRetriever(retriever=filter_retriever)

        result = multi_retriever.run(filters=filters)

        assert "documents" in result
        assert {doc.meta["lang"] for doc in result["documents"]} == expected_languages

    def test_deduplication_with_overlapping_results(self) -> None:
        doc1 = Document(content="Solar energy is renewable", id="doc1", score=0.9)
        doc2 = Document(content="Wind energy is clean", id="doc2", score=0.8)
        doc3 = Document(content="Solar energy is renewable", id="doc1", score=0.7)

        call_count = 0

        @component
        class MockRetriever:
            @component.output_types(documents=list[Document])
            def run(self, filters: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, list[Document]]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {"documents": [doc1, doc2]}
                return {"documents": [doc3, doc2]}

        multi_retriever = MultiFilterRetriever(retriever=MockRetriever(), max_workers=1)

        result = multi_retriever.run(
            filters=[
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {"field": "meta.lang", "operator": "==", "value": "de"},
            ]
        )

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert [doc.content for doc in result["documents"]].count("Solar energy is renewable") == 1
        assert [doc.content for doc in result["documents"]].count("Wind energy is clean") == 1

    def test_from_dict_roundtrip(self, in_memory_doc_store: InMemoryDocumentStore) -> None:
        filter_retriever = FilterRetriever(document_store=in_memory_doc_store)
        multi_retriever = MultiFilterRetriever(retriever=filter_retriever, max_workers=2)

        serialized = multi_retriever.to_dict()
        deserialized = MultiFilterRetriever.from_dict(serialized)

        assert isinstance(deserialized, MultiFilterRetriever)
        assert deserialized.max_workers == 2
