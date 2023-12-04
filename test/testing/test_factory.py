import pytest

from haystack.dataclasses import Document
from haystack.testing.factory import document_store_class, component_class
from haystack.core.component import component


def test_document_store_class_default():
    MyStore = document_store_class("MyStore")
    store = MyStore()
    assert store.count_documents() == 0
    assert store.filter_documents() == []
    assert store.write_documents([]) is None
    assert store.delete_documents([]) is None
    assert store.to_dict() == {"type": "haystack.testing.factory.MyStore", "init_parameters": {}}


def test_document_store_from_dict():
    MyStore = document_store_class("MyStore")

    store = MyStore.from_dict({"type": "haystack.testing.factory.MyStore", "init_parameters": {}})
    assert isinstance(store, MyStore)


def test_document_store_class_with_documents():
    doc = Document(id="fake_id", content="This is a document")
    MyStore = document_store_class("MyStore", documents=[doc])
    store = MyStore()
    assert store.count_documents() == 1
    assert store.filter_documents() == [doc]


def test_document_store_class_with_documents_count():
    MyStore = document_store_class("MyStore", documents_count=100)
    store = MyStore()
    assert store.count_documents() == 100
    assert store.filter_documents() == []


def test_document_store_class_with_documents_and_documents_count():
    doc = Document(id="fake_id", content="This is a document")
    MyStore = document_store_class("MyStore", documents=[doc], documents_count=100)
    store = MyStore()
    assert store.count_documents() == 100
    assert store.filter_documents() == [doc]


def test_document_store_class_with_bases():
    MyStore = document_store_class("MyStore", bases=(Exception,))
    store = MyStore()
    assert isinstance(store, Exception)


def test_document_store_class_with_extra_fields():
    MyStore = document_store_class("MyStore", extra_fields={"my_field": 10})
    store = MyStore()
    assert store.my_field == 10  # type: ignore


def test_component_class_default():
    MyComponent = component_class("MyComponent")
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": None}

    res = comp.run(value="something")
    assert res == {"value": None}

    res = comp.run(non_existing_input=1)
    assert res == {"value": None}


def test_component_class_is_registered():
    MyComponent = component_class("MyComponent")
    assert component.registry["haystack.testing.factory.MyComponent"] == MyComponent


def test_component_class_with_input_types():
    MyComponent = component_class("MyComponent", input_types={"value": int})
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": None}

    res = comp.run(value="something")
    assert res == {"value": None}


def test_component_class_with_output_types():
    MyComponent = component_class("MyComponent", output_types={"value": int})
    comp = MyComponent()

    res = comp.run(value=1)
    assert res == {"value": None}


def test_component_class_with_output():
    MyComponent = component_class("MyComponent", output={"value": 100})
    comp = MyComponent()
    res = comp.run(value=1)
    assert res == {"value": 100}


def test_component_class_with_output_and_output_types():
    MyComponent = component_class("MyComponent", output_types={"value": str}, output={"value": 100})
    comp = MyComponent()

    res = comp.run(value=1)
    assert res == {"value": 100}


def test_component_class_with_bases():
    MyComponent = component_class("MyComponent", bases=(Exception,))
    comp = MyComponent()
    assert isinstance(comp, Exception)


def test_component_class_with_extra_fields():
    MyComponent = component_class("MyComponent", extra_fields={"my_field": 10})
    comp = MyComponent()
    assert comp.my_field == 10  # type: ignore
