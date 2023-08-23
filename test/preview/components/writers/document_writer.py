from unittest.mock import MagicMock

import pytest

from haystack.preview import Document
from haystack.preview.components.writers.document_writer import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy
from test.preview.components.base import BaseTestComponent


class TestDocumentWriter(BaseTestComponent):
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
