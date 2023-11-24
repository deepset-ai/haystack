import pytest

from haystack.preview import Document, DeserializationError
from haystack.preview.testing.factory import document_store_class
from haystack.preview.document_stores.in_memory import InMemoryDocumentStore
from haystack.preview.components.caching.url_cache_checker import UrlCacheChecker


class TestUrlCacheChecker:
    @pytest.mark.unit
    def test_to_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = UrlCacheChecker(document_store=mocked_docstore_class())
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "url_field": "url",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = UrlCacheChecker(document_store=mocked_docstore_class(), url_field="my_url_field")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "url_field": "my_url_field",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        data = {
            "type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "url_field": "my_url_field",
            },
        }
        component = UrlCacheChecker.from_dict(data)
        assert isinstance(component.document_store, mocked_docstore_class)
        assert component.url_field == "my_url_field"

    @pytest.mark.unit
    def test_from_dict_without_docstore(self):
        data = {"type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            UrlCacheChecker.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_without_docstore_type(self):
        data = {
            "type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker",
            "init_parameters": {"document_store": {"init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="Missing 'type' in document store's serialization data"):
            UrlCacheChecker.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "haystack.preview.components.caching.url_cache_checker.UrlCacheChecker",
            "init_parameters": {"document_store": {"type": "NonexistingDocumentStore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="DocumentStore of type 'NonexistingDocumentStore' not found."):
            UrlCacheChecker.from_dict(data)

    @pytest.mark.unit
    def test_run(self):
        docstore = InMemoryDocumentStore()
        documents = [
            Document(content="doc1", meta={"url": "https://example.com/1"}),
            Document(content="doc2", meta={"url": "https://example.com/2"}),
            Document(content="doc3", meta={"url": "https://example.com/1"}),
            Document(content="doc4", meta={"url": "https://example.com/2"}),
        ]
        docstore.write_documents(documents)
        checker = UrlCacheChecker(docstore)
        results = checker.run(urls=["https://example.com/1", "https://example.com/5"])
        assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}
