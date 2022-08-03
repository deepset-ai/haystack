import sys
import logging

from unittest.mock import MagicMock

import pytest
import numpy as np

from haystack.document_stores.opensearch import (
    OpenSearch,
    OpenSearchDocumentStore,
    OpenDistroElasticsearchDocumentStore,
    RequestsHttpConnection,
    Urllib3HttpConnection,
    RequestError,
    tqdm,
)
from haystack.schema import Document, Label, Answer
from haystack.errors import DocumentStoreError

# Being all the tests in this module, ideally we wouldn't need a marker here,
# but this is to allow this test suite to be skipped when running (e.g.)
# `pytest test/document_stores --document-store-type=faiss`
class TestOpenSearchDocumentStore:

    # Constants

    query_emb = np.ndarray(shape=(2, 2), dtype=float)
    index_name = "myindex"

    # Fixtures

    @pytest.fixture
    def ds(self):
        """
        This fixture provides a working document store and takes care of removing the indices when done
        """
        index_name = __name__
        labels_index_name = f"{index_name}_labels"
        ds = OpenSearchDocumentStore(index=index_name, label_index=labels_index_name, port=9201, create_index=True)
        yield ds
        ds.delete_index(index_name)
        ds.delete_index(labels_index_name)

    @pytest.fixture
    def mocked_document_store(self):
        """
        The fixture provides an instance of a slightly customized
        OpenSearchDocumentStore equipped with a mocked client
        """

        class DSMock(OpenSearchDocumentStore):
            # We mock a subclass to avoid messing up the actual class object
            pass

        DSMock._init_client = MagicMock()
        DSMock.client = MagicMock()
        return DSMock()

    @pytest.fixture
    def mocked_open_search_init(self, monkeypatch):
        mocked_init = MagicMock(return_value=None)
        monkeypatch.setattr(OpenSearch, "__init__", mocked_init)
        return mocked_init

    @pytest.fixture
    def _init_client_params(self):
        """
        The fixture provides the required arguments to call OpenSearchDocumentStore._init_client
        """
        return {
            "host": "localhost",
            "port": 9999,
            "username": "user",
            "password": "pass",
            "aws4auth": None,
            "scheme": "http",
            "ca_certs": "ca_certs",
            "verify_certs": True,
            "timeout": 42,
            "use_system_proxy": True,
        }

    @pytest.fixture
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i} without embeddings",
                    meta={"name": f"name_{i}", "no_embedding": True, "month": "03"},
                )
            )

        return documents

    @pytest.fixture
    def index(self):
        return {
            "aliases": {},
            "mappings": {
                "properties": {
                    "age": {"type": "integer"},
                    "occupation": {"type": "text"},
                    "vec": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "engine": "nmslib",
                            "space_type": "innerproduct",
                            "name": "hnsw",
                            "parameters": {"ef_construction": 512, "m": 16},
                        },
                    },
                }
            },
            "settings": {
                "index": {
                    "creation_date": "1658337984559",
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                    "uuid": "jU5KPBtXQHOaIn2Cm2d4jg",
                    "version": {"created": "135238227"},
                    "provided_name": "fooindex",
                }
            },
        }

    @pytest.fixture
    def labels(self, documents):
        labels = []
        for i, d in enumerate(documents):
            labels.append(
                Label(
                    query="query",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    # create a mix set of labels
                    origin="user-feedback" if i % 2 else "gold-label",
                    answer=None if not i else Answer(f"the answer is {i}"),
                )
            )
        return labels

    # Integration tests

    @pytest.mark.integration
    def test___init__(self):
        OpenSearchDocumentStore(index="default_index", port=9201, create_index=True)

    @pytest.mark.integration
    def test_write_documents(self, ds, documents):
        ds.write_documents(documents)
        docs = ds.get_all_documents()
        assert len(docs) == len(documents)
        for i, doc in enumerate(docs):
            expected = documents[i]
            assert doc.id == expected.id

    @pytest.mark.integration
    def test_write_labels(self, ds, labels):
        ds.write_labels(labels)
        assert ds.get_all_labels() == labels

    @pytest.mark.integration
    def test_recreate_index(self, ds, documents, labels):
        ds.write_documents(documents)
        ds.write_labels(labels)

        # Create another document store on top of the previous one
        ds = OpenSearchDocumentStore(index=ds.index, label_index=ds.label_index, recreate_index=True, port=9201)
        assert len(ds.get_all_documents(index=ds.index)) == 0
        assert len(ds.get_all_labels(index=ds.label_index)) == 0

    @pytest.mark.integration
    def test_clone_embedding_field(self, ds, documents):
        cloned_field_name = "cloned"
        ds.write_documents(documents)
        ds.clone_embedding_field(cloned_field_name, "cosine")
        for doc in ds.get_all_documents():
            meta = doc.to_dict()["meta"]
            if "no_embedding" in meta:
                # docs with no embedding should be ignored
                assert cloned_field_name not in meta
            else:
                # docs with an original embedding should have the new one
                assert cloned_field_name in meta

    # Unit tests

    @pytest.mark.unit
    def test___init___api_key_raises_warning(self, mocked_document_store, caplog):
        with caplog.at_level(logging.WARN, logger="haystack.document_stores.opensearch"):
            mocked_document_store.__init__(api_key="foo")
            mocked_document_store.__init__(api_key_id="bar")
            mocked_document_store.__init__(api_key="foo", api_key_id="bar")

        assert len(caplog.records) == 3
        for r in caplog.records:
            assert r.levelname == "WARNING"

    @pytest.mark.unit
    def test___init___connection_test_fails(self, mocked_document_store):
        failing_client = MagicMock()
        failing_client.indices.get.side_effect = Exception("The client failed!")
        mocked_document_store._init_client.return_value = failing_client
        with pytest.raises(ConnectionError):
            mocked_document_store.__init__()

    @pytest.mark.unit
    def test___init___client_params(self, mocked_open_search_init, _init_client_params):
        """
        Ensure the Opensearch-py client was initialized with the right params
        """
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert mocked_open_search_init.called
        _, kwargs = mocked_open_search_init.call_args
        assert kwargs == {
            "hosts": [{"host": "localhost", "port": 9999}],
            "http_auth": ("user", "pass"),
            "scheme": "http",
            "ca_certs": "ca_certs",
            "verify_certs": True,
            "timeout": 42,
            "connection_class": RequestsHttpConnection,
        }

    @pytest.mark.unit
    def test__init_client_use_system_proxy_use_sys_proxy(self, mocked_open_search_init, _init_client_params):
        _init_client_params["use_system_proxy"] = False
        OpenSearchDocumentStore._init_client(**_init_client_params)
        _, kwargs = mocked_open_search_init.call_args
        assert kwargs["connection_class"] == Urllib3HttpConnection

    @pytest.mark.unit
    def test__init_client_use_system_proxy_dont_use_sys_proxy(self, mocked_open_search_init, _init_client_params):
        _init_client_params["use_system_proxy"] = True
        OpenSearchDocumentStore._init_client(**_init_client_params)
        _, kwargs = mocked_open_search_init.call_args
        assert kwargs["connection_class"] == RequestsHttpConnection

    @pytest.mark.unit
    def test__init_client_auth_methods_username_password(self, mocked_open_search_init, _init_client_params):
        _init_client_params["username"] = "user"
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        _, kwargs = mocked_open_search_init.call_args
        assert kwargs["http_auth"] == ("user", "pass")

    @pytest.mark.unit
    def test__init_client_auth_methods_aws_iam(self, mocked_open_search_init, _init_client_params):
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = "foo"
        OpenSearchDocumentStore._init_client(**_init_client_params)
        _, kwargs = mocked_open_search_init.call_args
        assert kwargs["http_auth"] == "foo"

    @pytest.mark.unit
    def test__init_client_auth_methods_no_auth(self, mocked_open_search_init, _init_client_params):
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        _, kwargs = mocked_open_search_init.call_args
        assert "http_auth" not in kwargs

    @pytest.mark.unit
    def test_query_by_embedding_raises_if_missing_field(self, mocked_document_store):
        mocked_document_store.embedding_field = ""
        with pytest.raises(DocumentStoreError):
            mocked_document_store.query_by_embedding(self.query_emb)

    @pytest.mark.unit
    def test_query_by_embedding_filters(self, mocked_document_store):
        expected_filters = {"type": "article", "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"}}
        mocked_document_store.query_by_embedding(self.query_emb, filters=expected_filters)
        # Assert the `search` method on the client was called with the filters we provided
        _, kwargs = mocked_document_store.client.search.call_args
        actual_filters = kwargs["body"]["query"]["bool"]["filter"]
        assert actual_filters["bool"]["must"] == [
            {"term": {"type": "article"}},
            {"range": {"date": {"gte": "2015-01-01", "lt": "2021-01-01"}}},
        ]

    @pytest.mark.unit
    def test_query_by_embedding_return_embedding_false(self, mocked_document_store):
        mocked_document_store.return_embedding = False
        mocked_document_store.query_by_embedding(self.query_emb)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["embedding"]}

    @pytest.mark.unit
    def test_query_by_embedding_excluded_meta_data_return_embedding_true(self, mocked_document_store):
        """
        Test that when `return_embedding==True` the field should NOT be excluded even if it
        was added to `excluded_meta_data`
        """
        mocked_document_store.return_embedding = True
        mocked_document_store.excluded_meta_data = ["foo", "embedding"]
        mocked_document_store.query_by_embedding(self.query_emb)
        _, kwargs = mocked_document_store.client.search.call_args
        # we expect "embedding" was removed from the final query
        assert kwargs["body"]["_source"] == {"excludes": ["foo"]}

    @pytest.mark.unit
    def test_query_by_embedding_excluded_meta_data_return_embedding_false(self, mocked_document_store):
        """
        Test that when `return_embedding==False`, the final query excludes the `embedding` field
        even if it wasn't explicitly added to `excluded_meta_data`
        """
        mocked_document_store.return_embedding = False
        mocked_document_store.excluded_meta_data = ["foo"]
        mocked_document_store.query_by_embedding(self.query_emb)
        # assert the resulting body is consistent with the `excluded_meta_data` value
        _, kwargs = mocked_document_store.client.search.call_args
        assert kwargs["body"]["_source"] == {"excludes": ["foo", "embedding"]}

    @pytest.mark.unit
    def test__create_document_index_with_alias(self, mocked_document_store, caplog):
        mocked_document_store.client.indices.exists_alias.return_value = True

        with caplog.at_level(logging.DEBUG, logger="haystack.document_stores.opensearch"):
            mocked_document_store._create_document_index(self.index_name)

        assert f"Index name {self.index_name} is an alias." in caplog.text

    @pytest.mark.unit
    def test__create_document_index_wrong_mapping_raises(self, mocked_document_store, index):
        """
        Ensure the method raises if we specify a field in `search_fields` that's not text
        """
        mocked_document_store.search_fields = ["age"]
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        with pytest.raises(Exception, match=f"The search_field 'age' of index '{self.index_name}' with type 'integer'"):
            mocked_document_store._create_document_index(self.index_name)

    @pytest.mark.unit
    def test__create_document_index_create_mapping_if_missing(self, mocked_document_store, index):
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "doesnt_have_a_mapping"

        mocked_document_store._create_document_index(self.index_name)

        # Assert the expected body was passed to the client
        _, kwargs = mocked_document_store.client.indices.put_mapping.call_args
        assert kwargs["index"] == self.index_name
        assert "doesnt_have_a_mapping" in kwargs["body"]["properties"]

    @pytest.mark.unit
    def test__create_document_index_with_bad_field_raises(self, mocked_document_store, index):
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "age"  # this is mapped as integer

        with pytest.raises(
            Exception, match=f"The '{self.index_name}' index in OpenSearch already has a field called 'age'"
        ):
            mocked_document_store._create_document_index(self.index_name)

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_but_no_method(self, mocked_document_store, index):
        """
        We call the method passing a properly mapped field but without the `method` specified in the mapping
        """
        del index["mappings"]["properties"]["vec"]["method"]
        # FIXME: the method assumes this key is present but it might not always be the case. This test has to pass
        # without the following line:
        index["settings"]["index"]["knn.space_type"] = "innerproduct"
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"

        mocked_document_store._create_document_index(self.index_name)
        # FIXME: when `method` is missing from the field mapping, embeddings_field_supports_similarity is always
        # False but I'm not sure this is by design
        assert mocked_document_store.embeddings_field_supports_similarity is False

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_similarity(self, mocked_document_store, index):
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.similarity = "dot_product"

        mocked_document_store._create_document_index(self.index_name)
        assert mocked_document_store.embeddings_field_supports_similarity is True

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_similarity_mismatch(
        self, mocked_document_store, index, caplog
    ):
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.similarity = "foo_bar"

        with caplog.at_level(logging.WARN, logger="haystack.document_stores.opensearch"):
            mocked_document_store._create_document_index(self.index_name)
        assert "Embedding field 'vec' is optimized for similarity 'dot_product'." in caplog.text
        assert mocked_document_store.embeddings_field_supports_similarity is False

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_adjust_params_hnsw_default(
        self, mocked_document_store, index
    ):
        """
        Test default values when `knn.algo_param` is missing from the index settings
        """
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.index_type = "hnsw"

        mocked_document_store._create_document_index(self.index_name)

        # assert the resulting body is contains the adjusted params
        _, kwargs = mocked_document_store.client.indices.put_settings.call_args
        assert kwargs["body"] == {"knn.algo_param.ef_search": 20}

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_adjust_params_hnsw(self, mocked_document_store, index):
        """
        Test a value of `knn.algo_param` that needs to be adjusted
        """
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.index_type = "hnsw"
        index["settings"]["index"]["knn.algo_param"] = {"ef_search": 999}

        mocked_document_store._create_document_index(self.index_name)

        # assert the resulting body is contains the adjusted params
        _, kwargs = mocked_document_store.client.indices.put_settings.call_args
        assert kwargs["body"] == {"knn.algo_param.ef_search": 20}

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_adjust_params_flat_default(
        self, mocked_document_store, index
    ):
        """
        If `knn.algo_param` is missing, default value needs no adjustments
        """
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.index_type = "flat"

        mocked_document_store._create_document_index(self.index_name)

        mocked_document_store.client.indices.put_settings.assert_not_called

    @pytest.mark.unit
    def test__create_document_index_with_existing_mapping_adjust_params_hnsw(self, mocked_document_store, index):
        """
        Test a value of `knn.algo_param` that needs to be adjusted
        """
        mocked_document_store.client.indices.exists.return_value = True
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.index_type = "flat"
        index["settings"]["index"]["knn.algo_param"] = {"ef_search": 999}

        mocked_document_store._create_document_index(self.index_name)

        # assert the resulting body is contains the adjusted params
        _, kwargs = mocked_document_store.client.indices.put_settings.call_args
        assert kwargs["body"] == {"knn.algo_param.ef_search": 512}

    @pytest.mark.unit
    def test__create_document_index_no_index_custom_mapping(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store.custom_mapping = {"mappings": {"properties": {"a_number": {"type": "integer"}}}}

        mocked_document_store._create_document_index(self.index_name)
        _, kwargs = mocked_document_store.client.indices.create.call_args
        assert kwargs["body"] == {"mappings": {"properties": {"a_number": {"type": "integer"}}}}

    @pytest.mark.unit
    def test__create_document_index_no_index_no_mapping(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store._create_document_index(self.index_name)
        _, kwargs = mocked_document_store.client.indices.create.call_args
        assert kwargs["body"] == {
            "mappings": {
                "dynamic_templates": [
                    {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string", "path_match": "*"}}
                ],
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "dimension": 768,
                        "method": {
                            "engine": "nmslib",
                            "name": "hnsw",
                            "parameters": {"ef_construction": 512, "m": 16},
                            "space_type": "innerproduct",
                        },
                        "type": "knn_vector",
                    },
                    "name": {"type": "keyword"},
                },
            },
            "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}, "index": {"knn": True}},
        }

    @pytest.mark.unit
    def test__create_document_index_no_index_no_mapping_with_synonyms(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store.search_fields = ["occupation"]
        mocked_document_store.synonyms = ["foo"]

        mocked_document_store._create_document_index(self.index_name)
        _, kwargs = mocked_document_store.client.indices.create.call_args
        assert kwargs["body"] == {
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "synonym"},
                    "occupation": {"type": "text", "analyzer": "synonym"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "space_type": "innerproduct",
                            "name": "hnsw",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 512, "m": 16},
                        },
                    },
                },
                "dynamic_templates": [
                    {"strings": {"path_match": "*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                ],
            },
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {"type": "standard"},
                        "synonym": {"tokenizer": "whitespace", "filter": ["lowercase", "synonym"]},
                    },
                    "filter": {"synonym": {"type": "synonym", "synonyms": ["foo"]}},
                },
                "index": {"knn": True},
            },
        }

    @pytest.mark.unit
    def test__create_document_index_no_index_no_mapping_with_embedding_field(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store.embedding_field = "vec"
        mocked_document_store.index_type = "hnsw"

        mocked_document_store._create_document_index(self.index_name)
        _, kwargs = mocked_document_store.client.indices.create.call_args
        assert kwargs["body"] == {
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "content": {"type": "text"},
                    "vec": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "space_type": "innerproduct",
                            "name": "hnsw",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 80, "m": 64},
                        },
                    },
                },
                "dynamic_templates": [
                    {"strings": {"path_match": "*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                ],
            },
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
                "index": {"knn": True, "knn.algo_param.ef_search": 20},
            },
        }

    @pytest.mark.unit
    def test__create_document_index_client_failure(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store.client.indices.create.side_effect = RequestError

        with pytest.raises(RequestError):
            mocked_document_store._create_document_index(self.index_name)

    @pytest.mark.unit
    def test__get_embedding_field_mapping_flat(self, mocked_document_store):
        mocked_document_store.index_type = "flat"

        assert mocked_document_store._get_embedding_field_mapping("dot_product") == {
            "type": "knn_vector",
            "dimension": 768,
            "method": {
                "space_type": "innerproduct",
                "name": "hnsw",
                "engine": "nmslib",
                "parameters": {"ef_construction": 512, "m": 16},
            },
        }

    @pytest.mark.unit
    def test__get_embedding_field_mapping_hnsw(self, mocked_document_store):
        mocked_document_store.index_type = "hnsw"

        assert mocked_document_store._get_embedding_field_mapping("dot_product") == {
            "type": "knn_vector",
            "dimension": 768,
            "method": {
                "space_type": "innerproduct",
                "name": "hnsw",
                "engine": "nmslib",
                "parameters": {"ef_construction": 80, "m": 64},
            },
        }

    @pytest.mark.unit
    def test__get_embedding_field_mapping_wrong(self, mocked_document_store, caplog):
        mocked_document_store.index_type = "foo"

        with caplog.at_level(logging.ERROR, logger="haystack.document_stores.opensearch"):
            retval = mocked_document_store._get_embedding_field_mapping("dot_product")

        assert "Please set index_type to either 'flat' or 'hnsw'" in caplog.text
        assert retval == {
            "type": "knn_vector",
            "dimension": 768,
            "method": {"space_type": "innerproduct", "name": "hnsw", "engine": "nmslib"},
        }

    @pytest.mark.unit
    def test__create_label_index_already_exists(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = True

        mocked_document_store._create_label_index("foo")
        mocked_document_store.client.indices.create.assert_not_called()

    @pytest.mark.unit
    def test__create_label_index_client_error(self, mocked_document_store):
        mocked_document_store.client.indices.exists.return_value = False
        mocked_document_store.client.indices.create.side_effect = RequestError

        with pytest.raises(RequestError):
            mocked_document_store._create_label_index("foo")

    @pytest.mark.unit
    def test__get_vector_similarity_query_support_true(self, mocked_document_store):
        mocked_document_store.embedding_field = "FooField"
        mocked_document_store.embeddings_field_supports_similarity = True

        assert mocked_document_store._get_vector_similarity_query(self.query_emb, 3) == {
            "bool": {"must": [{"knn": {"FooField": {"vector": self.query_emb.tolist(), "k": 3}}}]}
        }

    @pytest.mark.unit
    def test__get_vector_similarity_query_support_false(self, mocked_document_store):
        mocked_document_store.embedding_field = "FooField"
        mocked_document_store.embeddings_field_supports_similarity = False
        mocked_document_store.similarity = "dot_product"

        assert mocked_document_store._get_vector_similarity_query(self.query_emb, 3) == {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "FooField",
                        "query_value": self.query_emb.tolist(),
                        "space_type": "innerproduct",
                    },
                },
            }
        }

    @pytest.mark.unit
    def test__get_raw_similarity_score_dot(self, mocked_document_store):
        mocked_document_store.similarity = "dot_product"
        assert mocked_document_store._get_raw_similarity_score(2) == 1
        assert mocked_document_store._get_raw_similarity_score(-2) == 1.5

    @pytest.mark.unit
    def test__get_raw_similarity_score_l2(self, mocked_document_store):
        mocked_document_store.similarity = "l2"
        assert mocked_document_store._get_raw_similarity_score(1) == 0

    @pytest.mark.unit
    def test__get_raw_similarity_score_cosine(self, mocked_document_store):
        mocked_document_store.similarity = "cosine"
        mocked_document_store.embeddings_field_supports_similarity = True
        assert mocked_document_store._get_raw_similarity_score(1) == 1
        mocked_document_store.embeddings_field_supports_similarity = False
        assert mocked_document_store._get_raw_similarity_score(1) == 0

    @pytest.mark.unit
    def test_clone_embedding_field_duplicate_mapping(self, mocked_document_store, index):
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.index = self.index_name
        with pytest.raises(Exception, match="age already exists with mapping"):
            mocked_document_store.clone_embedding_field("age", "cosine")

    @pytest.mark.unit
    def test_clone_embedding_field_update_mapping(self, mocked_document_store, index, monkeypatch):
        mocked_document_store.client.indices.get.return_value = {self.index_name: index}
        mocked_document_store.index = self.index_name

        # Mock away tqdm and the batch logic so we can test the mapping update alone
        mocked_document_store._get_all_documents_in_index = MagicMock(return_value=[])
        monkeypatch.setattr(tqdm, "__new__", MagicMock())

        mocked_document_store.clone_embedding_field("a_field", "cosine")
        _, kwargs = mocked_document_store.client.indices.put_mapping.call_args
        assert kwargs["body"]["properties"]["a_field"] == {
            "type": "knn_vector",
            "dimension": 768,
            "method": {
                "space_type": "cosinesimil",
                "name": "hnsw",
                "engine": "nmslib",
                "parameters": {"ef_construction": 512, "m": 16},
            },
        }


class TestOpenDistroElasticsearchDocumentStore:
    @pytest.mark.unit
    def test_deprecation_notice(self, monkeypatch, caplog):
        klass = OpenDistroElasticsearchDocumentStore
        monkeypatch.setattr(klass, "_init_client", MagicMock())
        with caplog.at_level(logging.WARN, logger="haystack.document_stores.opensearch"):
            klass()
        assert caplog.record_tuples == [
            (
                "haystack.document_stores.opensearch",
                logging.WARN,
                "Open Distro for Elasticsearch has been replaced by OpenSearch! See https://opensearch.org/faq/ for details. We recommend using the OpenSearchDocumentStore instead.",
            )
        ]
