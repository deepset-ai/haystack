# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack import AsyncPipeline, DeserializationError, Pipeline
from haystack.components.retrievers.filter_retriever import FilterRetriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing.factory import document_store_class


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


class TestFilterRetrieverAsync:
    @classmethod
    def _documents_equal(cls, docs1: list[Document], docs2: list[Document]) -> bool:
        # # Order doesn't matter; we sort before comparing
        docs1.sort(key=lambda x: x.id)
        docs2.sort(key=lambda x: x.id)
        return docs1 == docs2

    @pytest.mark.asyncio
    async def test_retriever_init_filter(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "en"})
        result = await retriever.run_async()

        assert "documents" in result
        assert len(result["documents"]) == 3
        assert TestFilterRetrieverAsync._documents_equal(result["documents"], sample_docs["en_docs"])

    @pytest.mark.asyncio
    async def test_retriever_runtime_filter(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store)
        result = await retriever.run_async(filters={"field": "lang", "operator": "==", "value": "en"})

        assert "documents" in result
        assert len(result["documents"]) == 3
        assert TestFilterRetrieverAsync._documents_equal(result["documents"], sample_docs["en_docs"])

    @pytest.mark.asyncio
    async def test_retriever_init_filter_run_filter_override(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "en"})
        result = await retriever.run_async(filters={"field": "lang", "operator": "==", "value": "de"})

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert TestFilterRetrieverAsync._documents_equal(result["documents"], sample_docs["de_docs"])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_with_pipeline(self, sample_document_store, sample_docs):
        retriever = FilterRetriever(sample_document_store, filters={"field": "lang", "operator": "==", "value": "de"})

        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result: dict[str, Any] = await pipeline.run_async(data={"retriever": {}})

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert TestFilterRetrieverAsync._documents_equal(results_docs, sample_docs["de_docs"])

        result: dict[str, Any] = await pipeline.run_async(
            data={"retriever": {"filters": {"field": "lang", "operator": "==", "value": "en"}}}
        )

        assert result
        assert "retriever" in result
        results_docs = result["retriever"]["documents"]
        assert results_docs
        assert TestFilterRetrieverAsync._documents_equal(results_docs, sample_docs["en_docs"])
