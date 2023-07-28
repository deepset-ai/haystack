from unittest.mock import MagicMock

import pytest

from haystack.preview import Document
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy
from test.preview.components.base import BaseTestComponent


class TestDocumentWriter(BaseTestComponent):
    @pytest.mark.unit
    def test_writer_forwards_documents_and_policy_to_store(self):
        writer = DocumentWriter()
        documents = [
            Document(content="This is the text of a document."),
            Document(content="This is the text of another document."),
        ]

        mocked_store = MagicMock()
        mocked_store.__haystack_store__ = True
        writer.store = mocked_store
        writer.run(writer.input(documents=documents))

        # TODO check for default value DuplicatePolicy instead of None with new canals release
        writer.store.write_documents.assert_called_once_with(documents=documents, policy=None)

    @pytest.mark.unit
    def test_writer_fails_without_store(self):
        writer = DocumentWriter()
        documents = [Document(content="test")]
        with pytest.raises(
            ValueError,
            match="DocumentWriter needs a store to run: set the store instance to the " "self.store attribute",
        ):
            writer.run(writer.input(documents=documents))
