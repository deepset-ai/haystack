import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from elasticsearch import RequestError

from haystack.document_stores.elasticsearch8 import ElasticsearchDocumentStore, Elasticsearch
from haystack.document_stores.es_converter import elasticsearch_index_to_document_store
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes import PreProcessor
from haystack.testing import DocumentStoreBaseTestAbstract


class TestElasticsearchDocumentStore(DocumentStoreBaseTestAbstract):
    # Constants

    index_name = __name__
    query = "test"

    @pytest.fixture
    def ds(self):
        """
        This fixture provides a working document store and takes care of keeping clean
        the ES cluster used in the tests.
        """
        labels_index_name = f"{self.index_name}_labels"
        ds = ElasticsearchDocumentStore(
            index=self.index_name,
            label_index=labels_index_name,
            host=os.environ.get("ELASTICSEARCH_HOST", "localhost"),
            recreate_index=True,
        )

        yield ds

    @pytest.fixture
    def mocked_elastic_search_init(self, monkeypatch):
        mocked_init = MagicMock(return_value=None)
        monkeypatch.setattr(Elasticsearch, "__init__", mocked_init)
        return mocked_init

    @pytest.fixture
    def mocked_elastic_search_ping(self, monkeypatch):
        mocked_ping = MagicMock(return_value=True)
        monkeypatch.setattr(Elasticsearch, "ping", mocked_ping)
        return mocked_ping

    @pytest.fixture
    def mocked_document_store(self):
        """
        The fixture provides an instance of a slightly customized
        ElasticsearchDocumentStore equipped with a mocked client
        """

        class DSMock(ElasticsearchDocumentStore):
            # We mock a subclass to avoid messing up the actual class object
            pass

        DSMock._init_elastic_client = MagicMock()
        DSMock.client = MagicMock()
        return DSMock()

    @pytest.mark.integration
    def test___init__(self, ds):
        # defaults
        _ = ElasticsearchDocumentStore()

        # list of hosts + single port
        _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=9200)

        # list of hosts + list of ports (wrong)
        with pytest.raises(ValueError):
            _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200])

        # list of hosts + list
        _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200, 9200])

        # only api_key
        with pytest.raises(ValueError):
            _ = ElasticsearchDocumentStore(host=["localhost"], port=[9200], api_key="test")

        # api_key + id
        _ = ElasticsearchDocumentStore(host=["localhost"], port=[9200], api_key="test", api_key_id="test")

    @pytest.mark.integration
    def test_recreate_index(self, ds, documents, labels):
        ds.write_documents(documents)
        ds.write_labels(labels)

        # Create another document store on top of the previous one
        ds = ElasticsearchDocumentStore(index=ds.index, label_index=ds.label_index, recreate_index=True)
        assert len(ds.get_all_documents(index=ds.index)) == 0
        assert len(ds.get_all_labels(index=ds.label_index)) == 0

    @pytest.mark.integration
    def test_eq_filter(self, ds, documents):
        ds.write_documents(documents)

        filter = {"name": {"$eq": ["name_0"]}}
        filtered_docs = ds.get_all_documents(filters=filter)
        assert len(filtered_docs) == 3
        for doc in filtered_docs:
            assert doc.meta["name"] == "name_0"

        filter = {"numbers": {"$eq": [2, 4]}}
        filtered_docs = ds.query(query=None, filters=filter)
        assert len(filtered_docs) == 3
        for doc in filtered_docs:
            assert doc.meta["month"] == "01"
            assert doc.meta["numbers"] == [2, 4]

    @pytest.mark.integration
    def test_custom_fields(self, ds):
        index = "haystack_test_custom"
        document_store = ElasticsearchDocumentStore(
            index=index,
            content_field="custom_text_field",
            embedding_field="custom_embedding_field",
            recreate_index=True,
        )
        doc_to_write = {"custom_text_field": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
        document_store.write_documents([doc_to_write])
        documents = document_store.get_all_documents(return_embedding=True)
        assert len(documents) == 1
        assert documents[0].content == "test"
        np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)
        document_store.delete_index(index)

    @pytest.mark.integration
    def test_query_with_filters_and_missing_embeddings(self, ds, documents):
        ds.write_documents(documents)
        filters = {"month": {"$in": ["01", "03"]}}
        ds.skip_missing_embeddings = False
        with pytest.raises(RequestError):
            ds.query_by_embedding(np.random.rand(768), filters=filters)

        ds.skip_missing_embeddings = True
        documents = ds.query_by_embedding(np.random.rand(768), filters=filters)
        assert len(documents) == 3

    @pytest.mark.integration
    def test_synonyms(self, ds):
        synonyms = ["i-pod, i pod, ipod", "sea biscuit, sea biscit, seabiscuit", "foo, foo bar, baz"]
        synonym_type = "synonym_graph"

        client = ds.client
        index = "haystack_synonym_arg"
        client.options(ignore_status=[404]).indices.delete(index=index)
        ElasticsearchDocumentStore(index=index, synonyms=synonyms, synonym_type=synonym_type)
        indexed_settings = client.indices.get_settings(index=index)

        assert synonym_type == indexed_settings[index]["settings"]["index"]["analysis"]["filter"]["synonym"]["type"]
        assert synonyms == indexed_settings[index]["settings"]["index"]["analysis"]["filter"]["synonym"]["synonyms"]

    @pytest.mark.integration
    def test_search_field_mapping(self):
        index = "haystack_search_field_mapping"
        document_store = ElasticsearchDocumentStore(
            index=index, search_fields=["content", "sub_content"], content_field="title"
        )

        document_store.write_documents(
            [
                {
                    "title": "Green tea components",
                    "meta": {
                        "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
                        "sub_content": "Drink tip",
                    },
                    "id": "1",
                },
                {
                    "title": "Green tea catechin",
                    "meta": {
                        "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
                        "sub_content": "Ingredients tip",
                    },
                    "id": "2",
                },
                {
                    "title": "Minerals in Green tea",
                    "meta": {
                        "content": "Green tea also has small amounts of minerals that can benefit your health.",
                        "sub_content": "Minerals tip",
                    },
                    "id": "3",
                },
                {
                    "title": "Green tea Benefits",
                    "meta": {
                        "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
                        "sub_content": "Health tip",
                    },
                    "id": "4",
                },
            ]
        )

        indexed_settings = document_store.client.indices.get_mapping(index=index)

        assert indexed_settings[index]["mappings"]["properties"]["content"]["type"] == "text"
        assert indexed_settings[index]["mappings"]["properties"]["sub_content"]["type"] == "text"
        document_store.delete_index(index)

    @pytest.mark.integration
    def test_existing_alias(self, ds):
        client = ds.client
        client.options(ignore_status=[404]).indices.delete(index="haystack_existing_alias_1")
        client.options(ignore_status=[404]).indices.delete(index="haystack_existing_alias_2")
        client.options(ignore_status=[404]).indices.delete_alias(index="_all", name="haystack_existing_alias")

        settings = {"mappings": {"properties": {"content": {"type": "text"}}}}

        client.indices.create(index="haystack_existing_alias_1", **settings)
        client.indices.create(index="haystack_existing_alias_2", **settings)

        client.indices.put_alias(
            index="haystack_existing_alias_1,haystack_existing_alias_2", name="haystack_existing_alias"
        )

        # To be valid, all indices related to the alias must have content field of type text
        ElasticsearchDocumentStore(index="haystack_existing_alias", search_fields=["content"])

    @pytest.mark.integration
    def test_existing_alias_missing_fields(self, ds):
        client = ds.client
        client.options(ignore_status=[404]).indices.delete(index="haystack_existing_alias_1")
        client.options(ignore_status=[404]).indices.delete(index="haystack_existing_alias_2")
        client.options(ignore_status=[404]).indices.delete_alias(index="_all", name="haystack_existing_alias")

        right_settings = {"mappings": {"properties": {"content": {"type": "text"}}}}
        wrong_settings = {"mappings": {"properties": {"content": {"type": "histogram"}}}}

        client.indices.create(index="haystack_existing_alias_1", **right_settings)
        client.indices.create(index="haystack_existing_alias_2", **wrong_settings)
        client.indices.put_alias(
            index="haystack_existing_alias_1,haystack_existing_alias_2", name="haystack_existing_alias"
        )

        with pytest.raises(Exception):
            # wrong field type for "content" in index "haystack_existing_alias_2"
            ElasticsearchDocumentStore(
                index="haystack_existing_alias", search_fields=["content"], content_field="title"
            )

    @pytest.mark.integration
    def test_get_document_count_only_documents_without_embedding_arg(self, ds, documents):
        ds.write_documents(documents)

        assert ds.get_document_count() == 9
        assert ds.get_document_count(only_documents_without_embedding=True) == 3
        assert ds.get_document_count(only_documents_without_embedding=True, filters={"month": ["01"]}) == 0
        assert ds.get_document_count(only_documents_without_embedding=True, filters={"month": ["03"]}) == 3

    @pytest.mark.integration
    def test_elasticsearch_brownfield_support(self, ds, documents):
        ds.write_documents(documents)

        new_document_store = elasticsearch_index_to_document_store(
            document_store=InMemoryDocumentStore(),
            original_index_name=ds.index,
            original_content_field="content",
            original_name_field="name",
            included_metadata_fields=["date_field"],
            index="test_brownfield_support",
            id_hash_keys=["content", "meta"],
            verify_certs=False,
        )

        original_documents = ds.get_all_documents()
        transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support")
        assert len(original_documents) == len(transferred_documents)
        assert all("name" in doc.meta for doc in transferred_documents)
        assert all(doc.id == doc._get_id(["content", "meta"]) for doc in transferred_documents)

        original_content = set([doc.content for doc in original_documents])
        transferred_content = set([doc.content for doc in transferred_documents])
        assert original_content == transferred_content

        # Test transferring docs with PreProcessor
        new_document_store = elasticsearch_index_to_document_store(
            document_store=InMemoryDocumentStore(),
            original_index_name=ds.index,
            original_content_field="content",
            excluded_metadata_fields=["date_field"],
            index="test_brownfield_support_2",
            preprocessor=PreProcessor(split_length=1, split_respect_sentence_boundary=False),
            verify_certs=False,
        )
        transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support_2")
        assert all("name" in doc.meta for doc in transferred_documents)
        # Check if number of transferred_documents is equal to number of unique words.
        assert len(transferred_documents) == len(set(" ".join(original_content).split()))

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
    def test_get_document_by_id_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["source_excludes"] == "embedding"

    @pytest.mark.unit
    def test_get_document_by_id_excluded_meta_data_has_no_influence(self, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.return_embedding = False
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["source_excludes"] == "embedding"

    @pytest.mark.unit
    def test_write_documents_req_for_each_batch(self, mocked_document_store, documents):
        mocked_document_store.batch_size = 2
        with patch("haystack.document_stores.elasticsearch8.bulk") as mocked_bulk:
            mocked_document_store.write_documents(documents)
            assert mocked_bulk.call_count == 5

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
        assert kwargs["source_excludes"] == ["embedding"]

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.excluded_meta_data = ["foo", "embedding"]
        mocked_document_store.query(self.query)
        _, kwargs = mocked_document_store.client.options().search.call_args
        # we expect "embedding" was removed from the final query
        assert kwargs["source_excludes"] == ["foo"]

    @pytest.mark.unit
    def test_query_excluded_meta_data_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.query(self.query)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["source_excludes"] == ["foo", "embedding"]

    @pytest.mark.unit
    @patch("haystack.document_stores.elasticsearch8.scan")
    def test_get_all_documents_return_embedding_true(self, mocked_scan, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.get_all_documents(return_embedding=True)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_scan.call_args
        assert "_source" not in kwargs["query"]

    @pytest.mark.unit
    @patch("haystack.document_stores.elasticsearch8.scan")
    def test_get_all_documents_return_embedding_false(self, mocked_scan, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_scan.call_args
        assert kwargs["query"]["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    @patch("haystack.document_stores.elasticsearch8.scan")
    def test_get_all_documents_excluded_meta_data_has_no_influence(self, mocked_scan, mocked_document_store):
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.get_all_documents(return_embedding=False)
        # assert the resulting body is not affected by the `excluded_meta_data` value
        _, kwargs = mocked_scan.call_args
        assert kwargs["query"]["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_get_document_by_id_return_embedding_true(self, mocked_document_store):
        mocked_document_store.return_embedding = True
        mocked_document_store.get_document_by_id("123")
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.options().search.call_args
        assert kwargs["source_excludes"] is None

    @pytest.mark.unit
    @patch("haystack.document_stores.elasticsearch8.ElasticsearchDocumentStore._get_all_documents_in_index")
    def test_get_all_labels_legacy_document_id(self, mocked_get_all_documents_in_index, mocked_document_store):
        mocked_get_all_documents_in_index.return_value = [
            {
                "_id": "123",
                "_source": {
                    "query": "Who made the PDF specification?",
                    "document": {
                        "content": "Some content",
                        "content_type": "text",
                        "score": None,
                        "id": "fc18c987a8312e72a47fb1524f230bb0",
                        "meta": {},
                        "embedding": [0.1, 0.2, 0.3],
                    },
                    "answer": {
                        "answer": "Adobe Systems",
                        "type": "extractive",
                        "context": "Some content",
                        "offsets_in_context": [{"start": 60, "end": 73}],
                        "offsets_in_document": [{"start": 60, "end": 73}],
                        # legacy document_id answer
                        "document_id": "fc18c987a8312e72a47fb1524f230bb0",
                        "meta": {},
                        "score": None,
                    },
                    "is_correct_answer": True,
                    "is_correct_document": True,
                    "origin": "user-feedback",
                    "pipeline_id": "some-123",
                },
            }
        ]
        labels = mocked_document_store.get_all_labels()
        assert labels[0].answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]

    @pytest.mark.unit
    def test_query_batch_req_for_each_batch(self, mocked_document_store):
        mocked_document_store.batch_size = 2
        mocked_document_store.query_batch([self.query] * 3)
        assert mocked_document_store.client.msearch.call_count == 2

    @pytest.mark.unit
    def test_query_by_embedding_batch_req_for_each_batch(self, mocked_document_store):
        mocked_document_store.batch_size = 2
        mocked_document_store.query_by_embedding_batch([np.array([1, 2, 3])] * 3)
        assert mocked_document_store.client.msearch.call_count == 2
