import pytest
from haystack.document_stores.search_engine import SearchEngineDocumentStore, prepare_hosts


@pytest.mark.unit
def test_prepare_hosts():
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
    def test_get_meta_values_by_key(self, ds, documents):
        ds.write_documents(documents)

        # test without filters or query
        result = ds.get_metadata_values_by_key(key="name")
        assert result == [
            {"count": 3, "value": "name_0"},
            {"count": 3, "value": "name_1"},
            {"count": 3, "value": "name_2"},
        ]

        # test with filters but no query
        result = ds.get_metadata_values_by_key(key="year", filters={"month": ["01"]})
        assert result == [{"count": 3, "value": "2020"}]

        # test with filters & query
        result = ds.get_metadata_values_by_key(key="year", query="Bar")
        assert result == [{"count": 3, "value": "2021"}]


@pytest.mark.document_store
class TestSearchEngineDocumentStore:
    """
    This class tests the concrete methods in SearchEngineDocumentStore
    """

    @pytest.mark.integration
    def test__split_document_list(self):
        pass
