# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack import Document
from haystack.components.caching.cache_checker import CacheChecker
from haystack.testing.factory import document_store_class


class TestCacheCheckerAsync:
    @pytest.mark.asyncio
    async def test_run_async_invalid_docstore(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        checker = CacheChecker(document_store=mocked_docstore_class(), cache_field="url")

        with pytest.raises(TypeError, match="does not provide async support"):
            await checker.run_async(items=["https://example.com/1"])

    @pytest.mark.asyncio
    async def test_run_async(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
            Document(content="doc3", meta={"url": "https://example.com/1"}),
            Document(content="doc4", meta={"url": "https://example.com/2"}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="url")
        results = await checker.run_async(items=["https://example.com/1", "https://example.com/5"])
        assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}

    @pytest.mark.asyncio
    async def test_run_async_all_hits(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="url")
        results = await checker.run_async(items=["https://example.com/1", "https://example.com/2"])
        assert results["hits"] == documents
        assert results["misses"] == []

    @pytest.mark.asyncio
    async def test_run_async_all_misses(self, in_memory_doc_store):
        checker = CacheChecker(in_memory_doc_store, cache_field="url")
        results = await checker.run_async(items=["https://example.com/1", "https://example.com/2"])
        assert results["hits"] == []
        assert results["misses"] == ["https://example.com/1", "https://example.com/2"]

    @pytest.mark.asyncio
    async def test_run_async_filters_syntax(self):
        mock_store = MagicMock()
        mock_store.filter_documents_async = AsyncMock(return_value=[])
        checker = CacheChecker(document_store=mock_store, cache_field="url")
        await checker.run_async(items=["https://example.com/1"])
        expected_filters = {"field": "url", "operator": "==", "value": "https://example.com/1"}
        mock_store.filter_documents_async.assert_awaited_once_with(filters=expected_filters)
