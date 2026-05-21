# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from haystack import AsyncPipeline, Document, component
from haystack.components.retrievers import InMemoryEmbeddingRetriever, MultiQueryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class MockQueryEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        return {"embedding": np.ones(384).tolist()}

    @component.output_types(embedding=list[float])
    async def run_async(self, text: str) -> dict[str, list[float]]:
        return {"embedding": np.ones(384).tolist()}


class TestMultiQueryEmbeddingRetrieverAsync:
    @pytest.mark.asyncio
    async def test_run_async_with_empty_queries(self):
        multi_retriever = MultiQueryEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            query_embedder=MockQueryEmbedder(),
        )
        result = await multi_retriever.run_async(queries=[])
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
                self,
                query_embedding: list[float],
                filters: dict[str, Any] | None = None,
                top_k: int | None = None,
                **kwargs: Any,
            ) -> dict[str, list[Document]]:
                return {"documents": [doc_low, doc_high, doc_mid]}

            @component.output_types(documents=list[Document])
            async def run_async(
                self,
                query_embedding: list[float],
                filters: dict[str, Any] | None = None,
                top_k: int | None = None,
                **kwargs: Any,
            ) -> dict[str, list[Document]]:
                return {"documents": [doc_low, doc_high, doc_mid]}

        multi_retriever = MultiQueryEmbeddingRetriever(retriever=MockRetriever(), query_embedder=MockQueryEmbedder())
        result = await multi_retriever.run_async(queries=["query1", "query2"])

        scores = [doc.score for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_run_async_deduplication(self):
        doc2 = Document(content="Wind energy is clean", id="doc2", score=0.8)
        # doc3 shares the same id as doc1 — simulates the same doc retrieved by different queries
        doc3 = Document(content="Solar energy is renewable", id="doc1", score=0.7)

        @component
        class MockRetriever:
            @component.output_types(documents=list[Document])
            def run(
                self,
                query_embedding: list[float],
                filters: dict[str, Any] | None = None,
                top_k: int | None = None,
                **kwargs: Any,
            ) -> dict[str, list[Document]]:
                return {"documents": [doc3, doc2]}

            @component.output_types(documents=list[Document])
            async def run_async(
                self,
                query_embedding: list[float],
                filters: dict[str, Any] | None = None,
                top_k: int | None = None,
                **kwargs: Any,
            ) -> dict[str, list[Document]]:
                return {"documents": [doc3, doc2]}

        multi_retriever = MultiQueryEmbeddingRetriever(retriever=MockRetriever(), query_embedder=MockQueryEmbedder())
        result = await multi_retriever.run_async(queries=["query1", "query2"])

        assert "documents" in result
        assert len(result["documents"]) == 2
        contents = [doc.content for doc in result["documents"]]
        assert contents.count("Solar energy is renewable") == 1
        assert contents.count("Wind energy is clean") == 1

    @pytest.mark.asyncio
    async def test_run_async_falls_back_to_sync_when_no_run_async(self):
        @component
        class SyncOnlyEmbedder:
            @component.output_types(embedding=list[float])
            def run(self, text: str) -> dict[str, list[float]]:
                return {"embedding": np.ones(384).tolist()}

        multi_retriever = MultiQueryEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            query_embedder=SyncOnlyEmbedder(),
        )
        result = await multi_retriever.run_async(queries=["query"])
        assert "documents" in result
        assert result["documents"] == []

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_pipeline(self):
        multi_retriever = MultiQueryEmbeddingRetriever(
            retriever=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
            query_embedder=MockQueryEmbedder(),
        )
        pipeline = AsyncPipeline()
        pipeline.add_component("retriever", multi_retriever)
        result = await pipeline.run_async(data={"retriever": {"queries": ["green energy", "solar power"]}})

        assert result
        assert "retriever" in result
        assert "documents" in result["retriever"]
        assert result["retriever"]["documents"] == []
