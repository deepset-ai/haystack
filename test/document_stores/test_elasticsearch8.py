from unittest.mock import patch

import pytest

from test.document_stores.test_elasticsearch import ElasticsearchDocumentStoreTestAbstract


class TestElasticsearchDocumentStore8(ElasticsearchDocumentStoreTestAbstract):
    """
    This class tests the elasticsearch8.ElasticsearchDocumentStore. It modifies those tests from
    ElasticsearchDocumentStoreTestAbstract that test calling the client in the style of Elasticsearch 7.
    """

    @pytest.mark.unit
    def test_query_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert "_source" not in kwargs

    @pytest.mark.unit
    def test_query_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.excluded_meta_data = ["foo", "embedding"]
        mocked_document_store.query(self.query)
        _, kwargs = mocked_document_store.client.options().search.call_args
        # we expect "embedding" was removed from the final query
        assert kwargs["_source"] == {"excludes": ["foo"]}

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["_source"] == {"excludes": ["foo", "embedding"]}

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.client.options().search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=True)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert "_source" not in kwargs

    @pytest.mark.unit
    def test_get_all_documents_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.client.options().search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        # starting with elasticsearch client 7.16, scan() uses the query parameter instead of body,
        # see https://github.com/elastic/elasticsearch-py/commit/889edc9ad6d728b79fadf790238b79f36449d2e2
        body = kwargs.get("body", kwargs)
        assert body["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_all_documents_excluded_meta_data_has_no_influence(self, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.client.options().search.return_value = {}
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        # starting with elasticsearch client 7.16, scan() uses the query parameter instead of body,
        # see https://github.com/elastic/elasticsearch-py/commit/889edc9ad6d728b79fadf790238b79f36449d2e2
        body = kwargs.get("body", kwargs)
        assert body["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert "_source" not in kwargs

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_document_by_id_excluded_meta_data_has_no_influence(self, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_write_documents_req_for_each_batch(self, mocked_document_store, documents):
        mocked_document_store.batch_size = 2
        with patch("haystack.document_stores.elasticsearch8.bulk") as mocked_bulk:
            mocked_document_store.write_documents(documents)
            assert mocked_bulk.call_count == 5

    @pytest.mark.unit
    def test_import_from_haystack_document_stores_es8(self):
        from haystack.document_stores import ElasticsearchDocumentStore

        assert ElasticsearchDocumentStore.__module__ == "haystack.document_stores.elasticsearch8"
