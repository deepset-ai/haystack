import pytest
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore, MissingDocumentError, DuplicateDocumentError

#
# TODO make a base test class to test all future docdocstore against
#


class DocumentStoreBaseTests:
    @pytest.fixture
    def docstore(self):
        raise NotImplementedError()

    def direct_access(self, docstore, doc_id):
        """
        Bypass `filter_documents()`
        """
        raise NotImplementedError()

    def direct_write(self, docstore, documents):
        """
        Bypass `write_documents()`
        """
        raise NotImplementedError()

    def direct_delete(self, docstore, ids):
        """
        Bypass `delete_documents()`
        """
        raise NotImplementedError()

    def test_count_empty(self, docstore):
        assert docstore.count_documents() == 0

    def test_count_not_empty(self, docstore):
        self.direct_write(
            docstore, [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert docstore.count_documents() == 3

    def test_no_filter_empty(self, docstore):
        assert docstore.filter_documents() == []
        assert docstore.filter_documents(filters={}) == []

    def test_no_filter_not_empty(self, docstore):
        docs = [Document(content="test doc")]
        self.direct_write(docstore, docs)
        assert docstore.filter_documents() == docs
        assert docstore.filter_documents(filters={}) == docs

    #
    # TODO test filters
    #

    def test_write(self, docstore):
        doc = Document(content="test doc")
        docstore.write_documents(documents=[doc])
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_fail(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])
        with pytest.raises(DuplicateDocumentError, match=f"ID '{doc.id}' already exists."):
            docstore.write_documents(documents=[doc])
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_skip(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])
        docstore.write_documents(documents=[doc], duplicates="skip")
        assert self.direct_access(docstore, doc_id=doc.id) == doc

    def test_write_duplicate_overwrite(self, docstore):
        doc1 = Document(content="test doc 1")
        doc2 = Document(content="test doc 2")
        object.__setattr__(doc2, "id", doc1.id)  # Make two docs with different content but same ID

        self.direct_write(docstore, [doc2])
        assert self.direct_access(docstore, doc_id=doc1.id) == doc2
        docstore.write_documents(documents=[doc1], duplicates="overwrite")
        assert self.direct_access(docstore, doc_id=doc1.id) == doc1

    def test_write_not_docs(self, docstore):
        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            docstore.write_documents(["not a document for sure"])

    def test_write_not_list(self, docstore):
        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            docstore.write_documents("not a list actually")

    def test_delete_empty(self, docstore):
        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["test"])

    def test_delete_not_empty(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])

        docstore.delete_documents([doc.id])

        with pytest.raises(Exception):
            assert self.direct_access(docstore, doc_id=doc.id)

    def test_delete_not_empty_nonexisting(self, docstore):
        doc = Document(content="test doc")
        self.direct_write(docstore, [doc])

        with pytest.raises(MissingDocumentError):
            docstore.delete_documents(["non_existing"])

        assert self.direct_access(docstore, doc_id=doc.id) == doc
