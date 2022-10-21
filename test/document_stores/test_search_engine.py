import pytest
from haystack.document_stores.search_engine import SearchEngineDocumentStore, prepare_hosts


@pytest.mark.unit
def test_prepare_hosts(self):
    pass


@pytest.mark.document_store
class SearchEngineDocumentStoreTestAbstract:
    """
    This is the base class for any Searchengine Document Store testsuite, it doesn't have the `Test` prefix in the name
    because we want to run its methods only in subclasses.
    """

    @pytest.mark.integration
    def test___do_bulk(self):
        pass

    @pytest.mark.integration
    def test___do_scan(self):
        pass

    @pytest.mark.integration
    def test_query_by_embedding(self):
        pass

    @pytest.mark.integration
    def test_delete_index(self, ds):
        client = ds.client
        # the index should exist
        assert client.indices.exists(index=ds.index)
        ds.delete_index(ds.index)
        # the index was deleted and should not exist
        assert not client.indices.exists(index=ds.index)


@pytest.mark.document_store
class TestSearchEngineDocumentStore:
    """
    This class tests the concrete methods in SearchEngineDocumentStore
    """

    @pytest.mark.integration
    def test__split_document_list(self):
        pass
