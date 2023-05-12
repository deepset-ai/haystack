import pytest
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.document_stores._base import DocumentStoreBaseTests


class TestMemoryDocumentStore(DocumentStoreBaseTests):
    """
    Test MemoryDocumentStore's specific features
    """

    @pytest.fixture
    def docstore(self) -> MemoryDocumentStore:
        return MemoryDocumentStore()
