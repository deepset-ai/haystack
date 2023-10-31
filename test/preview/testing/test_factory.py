import pytest

from haystack.preview.dataclasses import Document
from haystack.preview.testing.factory import document_store_class
from haystack.preview.document_stores.decorator import document_store


@pytest.mark.unit
def test_document_store_class_default():
    MyStore = document_store_class("MyStore")
    store = MyStore()
    assert store.count_documents() == 0
    assert store.filter_documents() == []
    assert store.write_documents([]) is None
    assert store.delete_documents([]) is None
    assert store.to_dict() == {"type": "MyStore", "init_parameters": {}}


@pytest.mark.unit
def test_document_store_from_dict():
    MyStore = document_store_class("MyStore")

    store = MyStore.from_dict({"type": "MyStore", "init_parameters": {}})
    assert isinstance(store, MyStore)


@pytest.mark.unit
def test_document_store_class_is_registered():
    MyStore = document_store_class("MyStore")
    assert document_store.registry["MyStore"] == MyStore


@pytest.mark.unit
def test_document_store_class_with_documents():
    doc = Document(id="fake_id", content="This is a document")
    MyStore = document_store_class("MyStore", documents=[doc])
    store = MyStore()
    assert store.count_documents() == 1
    assert store.filter_documents() == [doc]


@pytest.mark.unit
def test_document_store_class_with_documents_count():
    MyStore = document_store_class("MyStore", documents_count=100)
    store = MyStore()
    assert store.count_documents() == 100
    assert store.filter_documents() == []


@pytest.mark.unit
def test_document_store_class_with_documents_and_documents_count():
    doc = Document(id="fake_id", content="This is a document")
    MyStore = document_store_class("MyStore", documents=[doc], documents_count=100)
    store = MyStore()
    assert store.count_documents() == 100
    assert store.filter_documents() == [doc]


@pytest.mark.unit
def test_document_store_class_with_bases():
    MyStore = document_store_class("MyStore", bases=(Exception,))
    store = MyStore()
    assert isinstance(store, Exception)


@pytest.mark.unit
def test_document_store_class_with_extra_fields():
    MyStore = document_store_class("MyStore", extra_fields={"my_field": 10})
    store = MyStore()
    assert store.my_field == 10
