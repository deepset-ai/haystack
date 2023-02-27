import logging
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore, Elasticsearch
from haystack.document_stores.es_converter import elasticsearch_index_to_document_store
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes import PreProcessor
from haystack.testing import DocumentStoreBaseTestAbstract

from .test_search_engine import SearchEngineDocumentStoreTestAbstract


class TestElasticsearchDocumentStore(DocumentStoreBaseTestAbstract, SearchEngineDocumentStoreTestAbstract):
    # Constants

    index_name = __name__

    @pytest.fixture
    def ds(self):
        """
        This fixture provides a working document store and takes care of removing the indices when done
        """
        labels_index_name = f"{self.index_name}_labels"
        ds = ElasticsearchDocumentStore(
            index=self.index_name,
            label_index=labels_index_name,
            host=os.environ.get("ELASTICSEARCH_HOST", "localhost"),
            create_index=True,
        )
        yield ds
        ds.delete_index(self.index_name)
        ds.delete_index(labels_index_name)

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
    def test___init__(self):
        # defaults
        _ = ElasticsearchDocumentStore()

        # list of hosts + single port
        _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=9200)

        # list of hosts + list of ports (wrong)
        with pytest.raises(Exception):
            _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200])

        # list of hosts + list
        _ = ElasticsearchDocumentStore(host=["localhost", "127.0.0.1"], port=[9200, 9200])

        # only api_key
        with pytest.raises(Exception):
            _ = ElasticsearchDocumentStore(host=["localhost"], port=[9200], api_key="test")

        # api_key +  id
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
        with pytest.raises(ds._RequestError):
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
        client.indices.delete(index=index, ignore=[404])
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
        client.indices.delete(index="haystack_existing_alias_1", ignore=[404])
        client.indices.delete(index="haystack_existing_alias_2", ignore=[404])
        client.indices.delete_alias(index="_all", name="haystack_existing_alias", ignore=[404])

        settings = {"mappings": {"properties": {"content": {"type": "text"}}}}

        client.indices.create(index="haystack_existing_alias_1", body=settings)
        client.indices.create(index="haystack_existing_alias_2", body=settings)

        client.indices.put_alias(
            index="haystack_existing_alias_1,haystack_existing_alias_2", name="haystack_existing_alias"
        )

        # To be valid, all indices related to the alias must have content field of type text
        ElasticsearchDocumentStore(index="haystack_existing_alias", search_fields=["content"])

    @pytest.mark.integration
    def test_existing_alias_missing_fields(self, ds):
        client = ds.client
        client.indices.delete(index="haystack_existing_alias_1", ignore=[404])
        client.indices.delete(index="haystack_existing_alias_2", ignore=[404])
        client.indices.delete_alias(index="_all", name="haystack_existing_alias", ignore=[404])

        right_settings = {"mappings": {"properties": {"content": {"type": "text"}}}}
        wrong_settings = {"mappings": {"properties": {"content": {"type": "histogram"}}}}

        client.indices.create(index="haystack_existing_alias_1", body=right_settings)
        client.indices.create(index="haystack_existing_alias_2", body=wrong_settings)
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
        )
        transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support_2")
        assert all("name" in doc.meta for doc in transferred_documents)
        # Check if number of transferred_documents is equal to number of unique words.
        assert len(transferred_documents) == len(set(" ".join(original_content).split()))

    @pytest.mark.unit
    def test__init_elastic_client_aws4auth_and_username_raises_warning(
        self, caplog, mocked_elastic_search_init, mocked_elastic_search_ping
    ):
        _init_client_remaining_kwargs = {
            "host": "host",
            "port": 443,
            "password": "pass",
            "api_key_id": None,
            "api_key": None,
            "scheme": "https",
            "ca_certs": None,
            "verify_certs": True,
            "timeout": 10,
            "use_system_proxy": False,
        }

        with caplog.at_level(logging.WARN, logger="haystack.document_stores.elasticsearch"):
            ElasticsearchDocumentStore._init_elastic_client(
                username="admin", aws4auth="foo", **_init_client_remaining_kwargs
            )
        assert len(caplog.records) == 1
        for r in caplog.records:
            assert r.levelname == "WARNING"

        caplog.clear()
        with caplog.at_level(logging.WARN, logger="haystack.document_stores.elasticsearch"):
            ElasticsearchDocumentStore._init_elastic_client(
                username=None, aws4auth="foo", **_init_client_remaining_kwargs
            )
            ElasticsearchDocumentStore._init_elastic_client(
                username="", aws4auth="foo", **_init_client_remaining_kwargs
            )
        assert len(caplog.records) == 0
