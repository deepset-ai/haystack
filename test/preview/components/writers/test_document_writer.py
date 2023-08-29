from unittest.mock import MagicMock

import pytest

from haystack.preview import Document, DeserializationError
from haystack.preview.document_stores import document_store
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy


class TestDocumentWriter:
    @pytest.mark.unit
    def test_to_dict(self):
        mocked_document_store = MagicMock()
        mocked_document_store.to_dict.return_value = {
            "type": "MockedDocumentStore",
            "init_parameters": {"parameter": 100},
        }
        component = DocumentWriter(document_store=mocked_document_store)
        data = component.to_dict()
        assert data == {
            "type": "DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "MockedDocumentStore", "init_parameters": {"parameter": 100}},
                "policy": "FAIL",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        mocked_document_store = MagicMock()
        mocked_document_store.to_dict.return_value = {
            "type": "MockedDocumentStore",
            "init_parameters": {"parameter": 100},
        }
        component = DocumentWriter(document_store=mocked_document_store, policy=DuplicatePolicy.SKIP)
        data = component.to_dict()
        assert data == {
            "type": "DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "MockedDocumentStore", "init_parameters": {"parameter": 100}},
                "policy": "SKIP",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        @document_store
        class MockedDocumentStore:
            def __init__(self, parameter):
                self.parameter = parameter

            def to_dict(self):
                return {"type": "MockedDocumentStore", "init_parameters": {"parameter": self.parameter}}

            @classmethod
            def from_dict(cls, data):
                return cls(parameter=data["init_parameters"]["parameter"])

        data = {
            "type": "DocumentWriter",
            "init_parameters": {
                "document_store": {"type": "MockedDocumentStore", "init_parameters": {"parameter": 100}},
                "policy": "SKIP",
            },
        }
        component = DocumentWriter.from_dict(data)
        assert isinstance(component.document_store, MockedDocumentStore)
        assert component.document_store.parameter == 100
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
