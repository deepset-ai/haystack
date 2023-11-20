from unittest.mock import MagicMock

import pytest

from haystack.preview import Document, DeserializationError
from haystack.preview.testing.factory import document_store_class
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy


class TestDocumentWriter:
    @pytest.mark.unit
    def test_to_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = DocumentWriter(document_store=mocked_docstore_class())
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "policy": "FAIL",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        component = DocumentWriter(document_store=mocked_docstore_class(), policy=DuplicatePolicy.SKIP)
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "policy": "SKIP",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        mocked_docstore_class = document_store_class("MockedDocumentStore")
        data = {
            "type": "haystack.preview.components.writers.document_writer.DocumentWriter",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.preview.testing.factory.MockedDocumentStore",
                    "init_parameters": {},
                },
                "policy": "SKIP",
            },
        }
        component = DocumentWriter.from_dict(data)
        assert isinstance(component.document_store, mocked_docstore_class)
        assert component.policy == DuplicatePolicy.SKIP

    @pytest.mark.unit
    def test_from_dict_without_docstore(self):
        data = {"type": "DocumentWriter", "init_parameters": {}}
        with pytest.raises(DeserializationError, match="Missing 'document_store' in serialization data"):
            DocumentWriter.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_without_docstore_type(self):
        data = {"type": "DocumentWriter", "init_parameters": {"document_store": {"init_parameters": {}}}}
        with pytest.raises(DeserializationError, match="Missing 'type' in document store's serialization data"):
            DocumentWriter.from_dict(data)

    @pytest.mark.unit
    def test_from_dict_nonexisting_docstore(self):
        data = {
            "type": "DocumentWriter",
            "init_parameters": {"document_store": {"type": "NonexistingDocumentStore", "init_parameters": {}}},
        }
        with pytest.raises(DeserializationError, match="DocumentStore of type 'NonexistingDocumentStore' not found."):
            DocumentWriter.from_dict(data)

    @pytest.mark.unit
    def test_run(self):
        mocked_document_store = MagicMock()
        writer = DocumentWriter(mocked_document_store)
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        writer.run(documents=documents)
        mocked_document_store.write_documents.assert_called_once_with(documents=documents, policy=DuplicatePolicy.FAIL)
