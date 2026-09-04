# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from haystack import Document
from haystack.components.caching.cache_checker import CacheChecker
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.testing.factory import document_store_class


@pytest.fixture()
def strict_datetime_doc_store():
    store = InMemoryDocumentStore(strict_datetime_comparison=True)
    yield store
    store.shutdown()


class TestCacheChecker:
    def test_to_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = CacheChecker(document_store=mocked_docstore_class(), cache_field="url")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {"type": "haystack.testing.factory.MockedDocumentStore", "init_parameters": {}},
                "cache_field": "url",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = CacheChecker(document_store=mocked_docstore_class(), cache_field="my_url_field")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {"type": "haystack.testing.factory.MockedDocumentStore", "init_parameters": {}},
                "cache_field": "my_url_field",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "cache_field": "my_url_field",
            },
        }
        component = CacheChecker.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.cache_field == "my_url_field"

    def test_from_dict_without_docstore(self):
        data = {"type": "haystack.components.caching.cache_checker.CacheChecker", "init_parameters": {}}
        with pytest.raises(
            TypeError, match="missing 2 required positional arguments: 'document_store' and 'cache_field'"
        ):
            CacheChecker.from_dict(data)

    def test_from_dict_nonexisting_docstore(self):
        # Use a type whose module passes the deserialization allowlist (haystack.*) but cannot be
        # resolved, so we still exercise the "import failed" code path rather than the allowlist gate.
        data = {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {"type": "haystack.does.not.exist.DocumentStore", "init_parameters": {}}
            },
        }
        with pytest.raises(
            ImportError, match=r"Failed to deserialize 'document_store':.*haystack\.does\.not\.exist\.DocumentStore"
        ):
            CacheChecker.from_dict(data)

    def test_run(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
            Document(content="doc3", meta={"url": "https://example.com/1"}),
            Document(content="doc4", meta={"url": "https://example.com/2"}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="url")
        results = checker.run(items=["https://example.com/1", "https://example.com/5"])
        assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}

    def test_filters_syntax(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        with patch.object(mocked_docstore_class, "filter_documents") as filter_documents:
            checker = CacheChecker(document_store=mocked_docstore_class(), cache_field="url")
            checker.run(items=["https://example.com/1", "https://example.com/2"])
            valid_filters_syntax = {
                "field": "url",
                "operator": "in",
                "value": ["https://example.com/1", "https://example.com/2"],
            }
            filter_documents.assert_any_call(filters=valid_filters_syntax)

    def test_run_queries_the_document_store_once(self, in_memory_doc_store):
        in_memory_doc_store.write_documents(
            [Document(content=f"doc{i}", meta={"url": f"https://example.com/{i}"}) for i in range(200)]
        )
        checker = CacheChecker(in_memory_doc_store, cache_field="url")

        with patch.object(
            in_memory_doc_store, "filter_documents", wraps=in_memory_doc_store.filter_documents
        ) as filter_documents:
            results = checker.run(items=[f"https://example.com/{i}" for i in range(200)])

        assert filter_documents.call_count == 1
        assert len(results["hits"]) == 200
        assert results["misses"] == []

    def test_run_with_no_items_does_not_query_the_document_store(self, in_memory_doc_store):
        checker = CacheChecker(in_memory_doc_store, cache_field="url")

        with patch.object(in_memory_doc_store, "filter_documents") as filter_documents:
            results = checker.run(items=[])

        assert filter_documents.call_count == 0
        assert results == {"hits": [], "misses": []}

    def test_run_repeats_hits_for_a_repeated_item(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="url")

        results = checker.run(items=["https://example.com/1", "https://example.com/1"])

        assert results["hits"] == [documents[0], documents[0]]
        assert results["misses"] == []

    def test_run_keeps_hits_grouped_by_item_and_misses_in_order(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="url")

        results = checker.run(
            items=[
                "https://example.com/2",
                "https://example.com/missing-b",
                "https://example.com/1",
                "https://example.com/missing-a",
            ]
        )

        assert results["hits"] == [documents[1], documents[0]]
        assert results["misses"] == ["https://example.com/missing-b", "https://example.com/missing-a"]

    def test_run_on_a_document_field_rather_than_a_meta_key(self, in_memory_doc_store):
        documents = [Document(content="doc1"), Document(content="doc2")]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="content")

        results = checker.run(items=["doc1", "doc3"])

        assert results["hits"] == [documents[0]]
        assert results["misses"] == ["doc3"]

    def test_run_on_a_nested_meta_field(self, in_memory_doc_store):
        documents = [
            Document(content="doc1", meta={"source": {"url": "https://example.com/1"}}),
            Document(content="doc2", meta={"source": {"url": "https://example.com/2"}}),
        ]
        in_memory_doc_store.write_documents(documents)
        checker = CacheChecker(in_memory_doc_store, cache_field="meta.source.url")

        results = checker.run(items=["https://example.com/2", "https://example.com/3"])

        assert results["hits"] == [documents[1]]
        assert results["misses"] == ["https://example.com/3"]

    def test_run_matches_datetimes_as_strictly_as_the_store_does(self, strict_datetime_doc_store):
        # The store was asked one `in` query and answered it with its own strictness, so a document whose
        # timestamp is timezone-aware comes back for the aware item alone. Grouping the answer onto items
        # has to compare the same way, or the naive item borrows the aware item's document.
        document = Document(content="doc1", meta={"fetched_at": "2026-01-01T12:00:00+00:00"})
        strict_datetime_doc_store.write_documents([document])
        checker = CacheChecker(strict_datetime_doc_store, cache_field="fetched_at")

        results = checker.run(items=["2026-01-01T12:00:00", "2026-01-01T12:00:00+00:00"])

        assert results["hits"] == [document]
        assert results["misses"] == ["2026-01-01T12:00:00"]

    def test_run_reconciles_datetimes_when_the_store_does(self, in_memory_doc_store):
        document = Document(content="doc1", meta={"fetched_at": "2026-01-01T12:00:00+00:00"})
        in_memory_doc_store.write_documents([document])
        checker = CacheChecker(in_memory_doc_store, cache_field="fetched_at")

        results = checker.run(items=["2026-01-01T12:00:00", "2026-01-01T12:00:00+00:00"])

        assert results["hits"] == [document, document]
        assert results["misses"] == []

    def test_close(self):
        closable_document_store = Mock(spec=["close"])
        checker = CacheChecker(document_store=closable_document_store, cache_field="url")
        checker.close()
        closable_document_store.close.assert_called_once_with()

        nonclosable_document_store = Mock(spec=[])
        checker = CacheChecker(document_store=nonclosable_document_store, cache_field="url")
        checker.close()
        assert nonclosable_document_store.mock_calls == []
