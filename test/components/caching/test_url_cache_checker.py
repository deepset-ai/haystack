import pytest

from haystack import Document, DeserializationError
from haystack.testing.factory import document_store_class
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.caching.cache_checker import CacheChecker


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
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            CacheChecker.from_dict(data)

    def test_from_dict_without_docstore_type(self):
        data = {
            "type": "haystack.components.caching.cache_checker.UrlCacheChecker",
            "init_parameters": {"document_store": {"init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="Missing 'type' in document store's serialization data"):
            CacheChecker.from_dict(data)

    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "haystack.components.caching.cache_checker.UrlCacheChecker",
            "init_parameters": {"document_store": {"type": "Nonexisting.DocumentStore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError):
            CacheChecker.from_dict(data)

    def test_run(self):
        docstore = InMemoryDocumentStore()
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
            Document(content="doc3", meta={"url": "https://example.com/1"}),
            Document(content="doc4", meta={"url": "https://example.com/2"}),
        ]
        docstore.write_documents(documents)
        checker = CacheChecker(docstore, cache_field="url")
        results = checker.run(items=["https://example.com/1", "https://example.com/5"])
        assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}
