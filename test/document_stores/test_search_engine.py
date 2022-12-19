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

    # Constants
    query = "test"

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

    @pytest.mark.unit
    def test_query_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert "_source" not in kwargs["body"]

    @pytest.mark.unit
    def test_query_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.excluded_meta_data = ["foo", "embedding"]
        mocked_document_store.query(self.query)
        _, kwargs = mocked_document_store.client.search.call_args
        # we expect "embedding" was removed from the final query
        assert kwargs["body"]["_source"] == {"excludes": ["foo"]}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["foo", "embedding"]}

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=True)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        # starting with elasticsearch client 7.16, scan() uses the query parameter instead of body,
        # see https://github.com/elastic/elasticsearch-py/commit/889edc9ad6d728b79fadf790238b79f36449d2e2
        body = kwargs.get("body", kwargs)
        assert "_source" not in body

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        # starting with elasticsearch client 7.16, scan() uses the query parameter instead of body,
        # see https://github.com/elastic/elasticsearch-py/commit/889edc9ad6d728b79fadf790238b79f36449d2e2
        body = kwargs.get("body", kwargs)
        assert body["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_all_documents_excluded_meta_data_has_no_influence(self, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.client.search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        # starting with elasticsearch client 7.16, scan() uses the query parameter instead of body,
        # see https://github.com/elastic/elasticsearch-py/commit/889edc9ad6d728b79fadf790238b79f36449d2e2
        body = kwargs.get("body", kwargs)
        assert body["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert "_source" not in kwargs["body"]

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_document_by_id_excluded_meta_data_has_no_influence(self, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["embedding"]}


@pytest.mark.document_store
class TestSearchEngineDocumentStore:
    """
    This class tests the concrete methods in SearchEngineDocumentStore
    """

    @pytest.mark.integration
    def test__split_document_list(self):
        pass
