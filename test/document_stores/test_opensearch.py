import sys

from unittest.mock import MagicMock

import pytest

from haystack.document_stores import OpenSearchDocumentStore, OpenDistroElasticsearchDocumentStore
from haystack.schema import Document, Label

from opensearchpy import OpenSearch, RequestsHttpConnection, Urllib3HttpConnection


# Skip OpenSearchDocumentStore tests on Windows
pytestmark = pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Opensearch not running on Windows CI")


class TestOpenSearchDocumentStore:
    @pytest.fixture
    def MockedOpenSearchDocumentStore(self, monkeypatch):
        """
        The fixture provides an OpenSearchDocumentStore
        equipped with a mocked client
        """
        klass = OpenSearchDocumentStore
        monkeypatch.setattr(klass, "_init_client", MagicMock())
        return klass

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
    def documents(self):
        return [
            Document(content="A Foo Document"),
            Document(content="A Bar Document"),
            Document(content="A Baz Document"),
        ]

    @pytest.mark.integration
    def test___init__(self):
        OpenSearchDocumentStore(index="default_index", port=9201, create_index=True)

    @pytest.mark.integration
    def test_write_documents_write_labels(self, document_store, documents):
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

        docs = document_store.get_all_documents()
        assert docs == documents

        lbls = document_store.get_all_labels()
        assert lbls == labels

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

    def test___init__api_key_raises_warning(self, MockedOpenSearchDocumentStore):
        with pytest.warns(UserWarning):
            MockedOpenSearchDocumentStore(api_key="foo")
            MockedOpenSearchDocumentStore(api_key_id="bar")
            MockedOpenSearchDocumentStore(api_key="foo", api_key_id="bar")

    def test___init__connection_test_fails(self, MockedOpenSearchDocumentStore):
        failing_client = MagicMock()
        failing_client.indices.get.side_effect = Exception("The client failed!")
        MockedOpenSearchDocumentStore._init_client.return_value = failing_client
        with pytest.raises(ConnectionError):
            MockedOpenSearchDocumentStore()

    def test__init_client_params(self, monkeypatch, _init_client_params):
        MockedOpenSearch = MagicMock()
        monkeypatch.setattr(OpenSearch, "__new__", MockedOpenSearch)
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert MockedOpenSearch.call_args.kwargs == {
            "hosts": [{"host": "localhost", "port": 9999}],
            "http_auth": ("user", "pass"),
            "scheme": "http",
            "ca_certs": "ca_certs",
            "verify_certs": True,
            "timeout": 42,
            "connection_class": RequestsHttpConnection,
        }

    def test__init_client_use_system_proxy(self, monkeypatch, _init_client_params):
        MockedOpenSearch = MagicMock()
        monkeypatch.setattr(OpenSearch, "__new__", MockedOpenSearch)

        _init_client_params["use_system_proxy"] = False
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert MockedOpenSearch.call_args.kwargs["connection_class"] == Urllib3HttpConnection

        _init_client_params["use_system_proxy"] = True
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert MockedOpenSearch.call_args.kwargs["connection_class"] == RequestsHttpConnection

    def test__init_client_auth_methods(self, monkeypatch, _init_client_params):
        MockedOpenSearch = MagicMock()
        monkeypatch.setattr(OpenSearch, "__new__", MockedOpenSearch)

        # Username/Password
        _init_client_params["username"] = "user"
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert MockedOpenSearch.call_args.kwargs["http_auth"] == ("user", "pass")

        # AWS IAM
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = "foo"
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert MockedOpenSearch.call_args.kwargs["http_auth"] == "foo"

        # No authentication
        _init_client_params["username"] = ""
        _init_client_params["aws4auth"] = None
        OpenSearchDocumentStore._init_client(**_init_client_params)
        assert "http_auth" not in MockedOpenSearch.call_args.kwargs


class TestOpenDistroElasticsearchDocumentStore:
    def test_deprecation_notice(self, monkeypatch):
        klass = OpenDistroElasticsearchDocumentStore
        monkeypatch.setattr(klass, "_init_client", MagicMock())
        with pytest.warns(UserWarning):
            klass()
