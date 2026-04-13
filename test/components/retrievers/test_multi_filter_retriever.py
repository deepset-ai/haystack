# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.retrievers.multi_filter_retriever import MultiFilterRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@pytest.fixture
def document_store() -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    DocumentWriter(document_store=store, policy=DuplicatePolicy.SKIP).run(
        documents=[
            Document(content="English text", id="doc1", meta={"lang": "en"}),
            Document(content="German text", id="doc2", meta={"lang": "de"}),
        ]
    )
    return store


class TestMultiFilterRetriever:
    def test_init(self, in_memory_doc_store) -> None:
        multi = MultiFilterRetriever(document_store=in_memory_doc_store)
        assert multi.document_store == in_memory_doc_store
        assert multi.max_workers == 3

    def test_init_custom_workers(self, in_memory_doc_store) -> None:
        multi = MultiFilterRetriever(document_store=in_memory_doc_store, max_workers=5)
        assert multi.max_workers == 5

    def test_run_empty_filters(self, document_store) -> None:
        multi = MultiFilterRetriever(document_store=document_store)
        assert multi.run(filters=[]) == {"documents": []}

    def test_run_single_filter(self, document_store) -> None:
        multi = MultiFilterRetriever(document_store=document_store)
        result = multi.run(filters=[{"field": "meta.lang", "operator": "==", "value": "en"}])
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["lang"] == "en"

    def test_run_multiple_filters(self, document_store) -> None:
        multi = MultiFilterRetriever(document_store=document_store)
        result = multi.run(
            filters=[
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {"field": "meta.lang", "operator": "==", "value": "de"},
            ]
        )
        assert len(result["documents"]) == 2
        assert {doc.meta["lang"] for doc in result["documents"]} == {"en", "de"}

    def test_deduplication(self) -> None:
        document_store = InMemoryDocumentStore()
        doc = Document(content="Haystack is awesome", id="doc", meta={"lang": "en", "type": "tech"})
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP).run(documents=[doc])

        multi = MultiFilterRetriever(document_store=document_store, max_workers=1)
        result = multi.run(
            filters=[
                {"field": "meta.lang", "operator": "==", "value": "en"},
                {"field": "meta.type", "operator": "==", "value": "tech"},
            ]
        )
        assert len(result["documents"]) == 1
        assert result["documents"][0].id == "doc"

    def test_to_dict(self, document_store) -> None:
        multi = MultiFilterRetriever(document_store=document_store, max_workers=2)
        data = component_to_dict(multi, "multi_filter")
        assert data["type"] == "haystack.components.retrievers.multi_filter_retriever.MultiFilterRetriever"
        assert data["init_parameters"]["max_workers"] == 2

    def test_from_dict(self, document_store) -> None:
        multi = MultiFilterRetriever(document_store=document_store, max_workers=2)
        serialized = component_to_dict(multi, "multi_filter")
        deserialized = component_from_dict(MultiFilterRetriever, serialized, "multi_filter")
        assert isinstance(deserialized, MultiFilterRetriever)
        assert deserialized.max_workers == 2
