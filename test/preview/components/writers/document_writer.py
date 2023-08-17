from unittest.mock import MagicMock

import pytest

from haystack.preview import Document
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy
from test.preview.components.base import BaseTestComponent


class TestDocumentWriter(BaseTestComponent):
    @pytest.mark.unit
    def test_run(self):
        writer = DocumentWriter()
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        mocked_document_store = MagicMock()
        mocked_document_store.__haystack_document_store__ = True
        writer.document_store = mocked_document_store
        writer.run(documents=documents)

        mocked_document_store.write_documents.assert_called_once_with(documents=documents, policy=DuplicatePolicy.FAIL)

    @pytest.mark.unit
    def test_run_without_store(self):
        writer = DocumentWriter()
        documents = [Document(content="test")]
        with pytest.raises(
            ValueError,
            match="DocumentWriter needs a DocumentStore to run: set the DocumentStore instance to the "
            "self.document_store attribute",
        ):
            writer.run(documents=documents)
