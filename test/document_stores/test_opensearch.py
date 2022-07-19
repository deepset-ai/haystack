import sys

from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import numpy as np

from haystack.document_stores.opensearch import (
    OpenSearch,
    OpenSearchDocumentStore,
    OpenDistroElasticsearchDocumentStore,
    RequestsHttpConnection,
    Urllib3HttpConnection,
)
from haystack.schema import Document, Label


# Skip OpenSearchDocumentStore tests on Windows
pytestmark = pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Opensearch not running on Windows CI")


class TestOpenSearchDocumentStore:

    # Constants

    query_emb = np.ndarray(shape=(2, 2), dtype=float)

    # Fixtures

    @pytest.fixture
    def document_store(self):
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

        # Mock a subclass to avoid messing up the actual class object
        class DSMock(OpenSearchDocumentStore):
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

    @pytest.fixture(scope="class")
    def documents(self):
        documents = []
        for i in range(3):
            i = 0
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        for i in range(3):
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    # Integration tests

    @pytest.mark.integration
    def test___init__(self):
        OpenSearchDocumentStore(index="default_index", port=9201, create_index=True)

    @pytest.mark.integration
    def test_write_documents(self, document_store, documents):
        document_store.write_documents(documents)
        docs = document_store.get_all_documents()
        assert len(docs) == len(documents)
        for i, doc in enumerate(docs):
            expected = documents[i]
            assert doc.id == expected.id

    @pytest.mark.integration
    def test_write_labels(self, document_store, documents):
        labels = []
        for d in documents:
            labels.append(
                Label(
                    query="query",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    origin="user-feedback",
                    answer=None,
                )
            )

        document_store.write_labels(labels)
        assert document_store.get_all_labels() == labels

    @pytest.mark.integration
    def test_recreate_index(self, document_store, documents):
        labels = []
        for d in documents:
            labels.append(
                Label(
                    query="query",
                    document=d,
                    is_correct_document=True,
                    is_correct_answer=False,
                    origin="user-feedback",
                    answer=None,
                )
            )

        document_store.write_documents(documents)
        document_store.write_labels(labels)

        # Create another document store on top of the previous one
        ds = OpenSearchDocumentStore(
            index=document_store.index, label_index=document_store.label_index, recreate_index=True, port=9201
        )
        assert len(ds.get_all_documents(index=document_store.index)) == 0
        assert len(ds.get_all_labels(index=document_store.label_index)) == 0

    # Unit tests

    def test_query_by_embedding_raises_if_missing_field(self, mocked_document_store):
        mocked_document_store.embedding_field = ""
        with pytest.raises(RuntimeError):
            mocked_document_store.query_by_embedding(self.query_emb)

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

    def test___init___api_key_raises_warning(self, mocked_document_store):
        with pytest.warns(UserWarning):
            mocked_document_store.__init__(api_key="foo")
            mocked_document_store.__init__(api_key_id="bar")
            mocked_document_store.__init__(api_key="foo", api_key_id="bar")

    def test___init___connection_test_fails(self, mocked_document_store):
        failing_client = MagicMock()
        failing_client.indices.get.side_effect = Exception("The client failed!")
        mocked_document_store._init_client.return_value = failing_client
        with pytest.raises(ConnectionError):
            mocked_document_store.__init__()

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

    def test__init_client_use_system_proxy(self, mocked_open_search_init, _init_client_params):
        _init_client_params["use_system_proxy"] = False
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert mocked_open_search_init.call_args.kwargs["connection_class"] == Urllib3HttpConnection

        _init_client_params["use_system_proxy"] = True
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert mocked_open_search_init.call_args.kwargs["connection_class"] == RequestsHttpConnection

    def test__init_client_auth_methods(self, mocked_open_search_init, _init_client_params):
        # Username/Password
        _init_client_params["username"] = "user"
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert mocked_open_search_init.call_args.kwargs["http_auth"] == ("user", "pass")

        # AWS IAM
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = "foo"
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert mocked_open_search_init.call_args.kwargs["http_auth"] == "foo"

        # No authentication
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert "http_auth" not in mocked_open_search_init.call_args.kwargs


class TestOpenDistroElasticsearchDocumentStore:
    def test_deprecation_notice(self, monkeypatch):
        klass = OpenDistroElasticsearchDocumentStore
        monkeypatch.setattr(klass, "_init_client", MagicMock())
        with pytest.warns(UserWarning):
            klass()
