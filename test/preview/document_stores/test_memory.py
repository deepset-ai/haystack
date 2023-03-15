import pytest
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import MemoryDocumentStore, MissingDocumentError, DuplicateDocumentError

#
# TODO make a base test class to test all future docstore against
#


def test_count_empty():
    store = MemoryDocumentStore()
    assert store.count_documents() == 0


def test_count_not_empty():
    store = MemoryDocumentStore()
    store.storage["test1"] = Document(content="test doc")
    store.storage["test2"] = Document(content="test doc")
    store.storage["test3"] = Document(content="test doc")
    assert store.count_documents() == 3


def test_store_no_filter_empty():
    store = MemoryDocumentStore()
    assert store.filter_documents() == []
    assert store.filter_documents(filters={}) == []


def test_store_no_filter_not_empty():
    store = MemoryDocumentStore()
    store.storage["test"] = Document(content="test doc")
    assert store.filter_documents() == [Document(content="test doc")]
    assert store.filter_documents(filters={}) == [Document(content="test doc")]


#
# TODO test filters when they will be implemented
#


def test_store_write():
    store = MemoryDocumentStore()
    doc = Document(content="test doc")
    store.write_documents(documents=[doc])
    assert doc.id in store.storage.keys()


def test_store_write_duplicate_fail():
    store = MemoryDocumentStore()
    doc = Document(content="test doc")
    store.storage[doc.id] = doc
    with pytest.raises(DuplicateDocumentError, match=f"ID '{doc.id}' already exists."):
        store.write_documents(documents=[doc])
    assert store.storage[doc.id] == doc


def test_store_write_duplicate_skip():
    store = MemoryDocumentStore()
    doc = Document(content="test doc")
    store.storage[doc.id] = doc
    store.write_documents(documents=[doc], duplicates="skip")
    assert store.storage[doc.id] == doc


def test_store_write_duplicate_overwrite():
    store = MemoryDocumentStore()
    doc1 = Document(content="test doc")
    doc2 = Document(content="test doc")
    store.storage[doc1.id] = doc2
    store.write_documents(documents=[doc1], duplicates="overwrite")
    assert store.storage[doc1.id] == doc1


def test_store_write_not_docs():
    store = MemoryDocumentStore()
    with pytest.raises(ValueError, match="Please provide a list of Documents"):
        store.write_documents(["not a document for sure"])


def test_store_write_not_list():
    store = MemoryDocumentStore()
    with pytest.raises(ValueError, match="Please provide a list of Documents"):
        store.write_documents("not a list actually")


def test_store_delete_empty():
    store = MemoryDocumentStore()
    with pytest.raises(MissingDocumentError):
        store.delete_documents(["test"])


def test_store_delete_not_empty():
    store = MemoryDocumentStore()
    store.storage["test"] = Document(content="test doc")
    store.delete_documents(["test"])
    assert "test" not in store.storage.keys()


def test_store_delete_not_empty_nonexisting():
    store = MemoryDocumentStore()
    store.storage["test"] = Document(content="test doc")
    with pytest.raises(MissingDocumentError):
        store.delete_documents(["non_existing"])
    assert "test" in store.storage.keys()
