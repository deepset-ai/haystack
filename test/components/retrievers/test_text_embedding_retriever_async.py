# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from haystack import AsyncPipeline, Document, component
from haystack.components.retrievers import InMemoryEmbeddingRetriever, TextEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class MockTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        return {"embedding": np.ones(384).tolist()}

    @component.output_types(embedding=list[float])
    async def run_async(self, text: str) -> dict[str, list[float]]:
        return {"embedding": np.ones(384).tolist()}


class TestTextEmbeddingRetrieverAsync:
    @pytest.mark.asyncio
    async def test_run_async_with_empty_document_store(self):
        retriever = TextEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            text_embedder=MockTextEmbedder(),
        )
        result = await retriever.run_async(query="green energy")
        assert "documents" in result
        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_run_async_returns_documents_sorted_by_score(self):
        doc_high = Document(content="Solar energy", id="doc1", score=0.9)
        doc_low = Document(content="Fossil fuels", id="doc2", score=0.3)
        doc_mid = Document(content="Wind energy", id="doc3", score=0.6)

        @component
        class MockRetriever:
            @component.output_types(documents=list[Document])
            def run(
                self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int | None = None
            ) -> dict[str, list[Document]]:
                return {"documents": [doc_low, doc_high, doc_mid]}

            @component.output_types(documents=list[Document])
            async def run_async(
                self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int | None = None
            ) -> dict[str, list[Document]]:
                return {"documents": [doc_low, doc_high, doc_mid]}

        retriever = TextEmbeddingRetriever(retriever=MockRetriever(), text_embedder=MockTextEmbedder())
        result = await retriever.run_async(query="energy")

        scores = [doc.score for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_run_async_falls_back_to_sync_when_no_run_async(self):
        @component
        class SyncOnlyEmbedder:
            @component.output_types(embedding=list[float])
            def run(self, text: str) -> dict[str, list[float]]:
                return {"embedding": np.ones(384).tolist()}

        retriever = TextEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            text_embedder=SyncOnlyEmbedder(),
        )
        result = await retriever.run_async(query="green energy")
        assert "documents" in result
        assert result["documents"] == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_pipeline(self):
        retriever = TextEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            text_embedder=MockTextEmbedder(),
        )
        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", retriever)
        result = await pipeline.run_async(data={"retriever": {"query": "green energy"}})

        assert result
        assert "retriever" in result
        assert "documents" in result["retriever"]
        assert result["retriever"]["documents"] == []
