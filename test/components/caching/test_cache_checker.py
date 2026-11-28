# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from haystack import Document
from haystack.components.caching.cache_checker import CacheChecker
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.errors import FilterError
from haystack.testing.factory import document_store_class


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
                "ttl": None,
                "time_field": "cached_at",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = CacheChecker(
            document_store=mocked_docstore_class(),
            cache_field="my_url_field",
            ttl=timedelta(hours=1),
            time_field="my_time_field",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {"type": "haystack.testing.factory.MockedDocumentStore", "init_parameters": {}},
                "cache_field": "my_url_field",
                "ttl": 3600.0,
                "time_field": "my_time_field",
            },
        }

    def test_to_dict_with_numeric_ttl(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = CacheChecker(document_store=mocked_docstore_class(), cache_field="url", ttl=90)
        data = component.to_dict()
        assert data["init_parameters"]["ttl"] == 90.0

    def test_from_dict(self):
        data = {
            "type": "haystack.components.caching.cache_checker.CacheChecker",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {},
                },
                "cache_field": "my_url_field",
                "ttl": 3600.0,
                "time_field": "my_time_field",
            },
        }
        component = CacheChecker.from_dict(data)
        assert isinstance(component.document_store, InMemoryDocumentStore)
        assert component.cache_field == "my_url_field"
        assert component.ttl == timedelta(hours=1)
        assert component.time_field == "my_time_field"

    def test_from_dict_without_ttl_and_time_field(self):
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
        assert component.ttl is None
        assert component.time_field == "cached_at"

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
            checker.run(items=["https://example.com/1"])
            valid_filters_syntax = {"field": "url", "operator": "==", "value": "https://example.com/1"}
            filter_documents.assert_any_call(filters=valid_filters_syntax)

    def test_close(self):
        closable_document_store = Mock(spec=["close"])
        checker = CacheChecker(document_store=closable_document_store, cache_field="url")
        checker.close()
        closable_document_store.close.assert_called_once_with()

        nonclosable_document_store = Mock(spec=[])
        checker = CacheChecker(document_store=nonclosable_document_store, cache_field="url")
        checker.close()
        assert nonclosable_document_store.mock_calls == []

    def test_run_with_ttl_fresh_hit(self, in_memory_doc_store):
        fresh_doc = Document(
            content="doc1", meta={"url": "https://example.com/1", "cached_at": datetime.now(UTC) - timedelta(minutes=5)}
        )
        in_memory_doc_store.write_documents([fresh_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [fresh_doc], "misses": []}

    def test_run_with_ttl_expired_is_miss(self, in_memory_doc_store):
        stale_doc = Document(
            content="doc1", meta={"url": "https://example.com/1", "cached_at": datetime.now(UTC) - timedelta(hours=2)}
        )
        in_memory_doc_store.write_documents([stale_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [], "misses": ["https://example.com/1"]}

    def test_run_with_ttl_missing_time_field_is_miss(self, in_memory_doc_store):
        undated_doc = Document(content="doc1", meta={"url": "https://example.com/1"})
        in_memory_doc_store.write_documents([undated_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [], "misses": ["https://example.com/1"]}

    def test_run_with_ttl_iso_string_timestamp(self, in_memory_doc_store):
        fresh_doc = Document(
            content="doc1",
            meta={"url": "https://example.com/1", "cached_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat()},
        )
        in_memory_doc_store.write_documents([fresh_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [fresh_doc], "misses": []}

    def test_run_without_ttl_ignores_time_field(self, in_memory_doc_store):
        # backward compatibility: no ttl configured means entries never expire, regardless of cached_at
        old_doc = Document(
            content="doc1", meta={"url": "https://example.com/1", "cached_at": datetime.now(UTC) - timedelta(days=365)}
        )
        in_memory_doc_store.write_documents([old_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url")
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [old_doc], "misses": []}

    def test_run_with_ttl_non_datetime_timestamp_raises_error(self, in_memory_doc_store):
        bad_doc = Document(content="doc1", meta={"url": "https://example.com/1", "cached_at": 12345})
        in_memory_doc_store.write_documents([bad_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        with pytest.raises(FilterError):
            checker.run(items=["https://example.com/1"])

    def test_ttl_accepts_numeric_seconds(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        checker = CacheChecker(document_store=mocked_docstore_class(), cache_field="url", ttl=3600)
        assert checker.ttl == timedelta(hours=1)

    @pytest.mark.parametrize("ttl", [0, -1, timedelta(0), timedelta(seconds=-1), "3600", True, float("inf")])
    def test_init_with_invalid_ttl(self, ttl):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        with pytest.raises(ValueError):
            CacheChecker(document_store=mocked_docstore_class(), cache_field="url", ttl=ttl)

    def test_time_field_with_meta_prefix(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        checker = CacheChecker(document_store=mocked_docstore_class(), cache_field="url", time_field="meta.cached_at")
        assert checker.time_field == "cached_at"

    def test_run_with_ttl_future_timestamp_is_miss(self, in_memory_doc_store):
        future_doc = Document(
            content="doc1", meta={"url": "https://example.com/1", "cached_at": datetime.now(UTC) + timedelta(hours=1)}
        )
        in_memory_doc_store.write_documents([future_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        results = checker.run(items=["https://example.com/1"])
        assert results == {"hits": [], "misses": ["https://example.com/1"]}

    def test_run_with_ttl_malformed_iso_string_raises_error(self, in_memory_doc_store):
        bad_doc = Document(content="doc1", meta={"url": "https://example.com/1", "cached_at": "not-a-timestamp"})
        in_memory_doc_store.write_documents([bad_doc])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        with pytest.raises(FilterError):
            checker.run(items=["https://example.com/1"])

    def test_build_filters_with_ttl(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        checker = CacheChecker(mocked_docstore_class(), cache_field="url", ttl=timedelta(hours=1))
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        assert checker._build_filters("https://example.com/1", now) == {
            "operator": "AND",
            "conditions": [
                {"field": "url", "operator": "==", "value": "https://example.com/1"},
                {"field": "meta.cached_at", "operator": ">=", "value": (now - timedelta(hours=1)).isoformat()},
                {"field": "meta.cached_at", "operator": "<=", "value": now.isoformat()},
            ],
        }

    def test_build_filters_with_ttl_and_meta_prefixed_time_field(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        checker = CacheChecker(
            mocked_docstore_class(), cache_field="url", ttl=timedelta(hours=1), time_field="meta.cached_at"
        )
        now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        time_conditions = checker._build_filters("https://example.com/1", now)["conditions"][1:]
        assert [condition["field"] for condition in time_conditions] == ["meta.cached_at", "meta.cached_at"]

    def test_run_with_ttl_stale_and_fresh_copies(self, in_memory_doc_store):
        url = "https://example.com/1"
        stale = Document(content="old", meta={"url": url, "cached_at": datetime.now(UTC) - timedelta(hours=2)})
        fresh = Document(content="new", meta={"url": url, "cached_at": datetime.now(UTC) - timedelta(minutes=5)})
        in_memory_doc_store.write_documents([stale, fresh])
        checker = CacheChecker(in_memory_doc_store, cache_field="url", ttl=timedelta(hours=1))
        assert checker.run(items=[url]) == {"hits": [fresh], "misses": []}
