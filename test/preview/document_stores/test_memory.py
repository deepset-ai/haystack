import pytest
from haystack.preview.document_stores import MemoryDocumentStore

from test.preview.document_stores._base import DocumentStoreBaseTests


class TestMemoryDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def docstore(self):
        return MemoryDocumentStore()

    def direct_access(self, docstore, doc_id):
        """
        Bypass `filter_documents()`
        """
        return docstore.storage[doc_id]

    def direct_write(self, docstore, documents):
        """
        Bypass `write_documents()`
        """
        for doc in documents:
            docstore.storage[doc.id] = doc

    def direct_delete(self, docstore, ids):
        """
        Bypass `delete_documents()`
        """
        for doc_id in ids:
            del docstore.storage[doc_id]

    #
    # Test retrieval
    #
