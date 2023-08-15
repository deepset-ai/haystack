from unittest.mock import Mock

import pytest

from haystack.preview.testing.factory import document_store_class
from haystack.preview.document_stores.decorator import default_document_store_to_dict, default_document_store_from_dict
from haystack.preview.document_stores.errors import DocumentStoreDeserializationError


@pytest.mark.unit
def test_default_store_to_dict():
    MyStore = document_store_class("MyStore")
    comp = MyStore()
    res = default_document_store_to_dict(comp)
    assert res == {"hash": id(comp), "type": "MyStore", "init_parameters": {}}


@pytest.mark.unit
def test_default_store_to_dict_with_custom_init_parameters():
    extra_fields = {"init_parameters": {"custom_param": True}}
    MyStore = document_store_class("MyStore", extra_fields=extra_fields)
    comp = MyStore()
    res = default_document_store_to_dict(comp)
    assert res == {"hash": id(comp), "type": "MyStore", "init_parameters": {"custom_param": True}}


@pytest.mark.unit
def test_default_store_from_dict():
    MyStore = document_store_class("MyStore")
    comp = default_document_store_from_dict(MyStore, {"type": "MyStore"})
    assert isinstance(comp, MyStore)


@pytest.mark.unit
def test_default_store_from_dict_with_custom_init_parameters():
    def store_init(self, custom_param: int):
        self.custom_param = custom_param

    extra_fields = {"__init__": store_init}
    MyStore = document_store_class("MyStore", extra_fields=extra_fields)
    comp = default_document_store_from_dict(MyStore, {"type": "MyStore", "init_parameters": {"custom_param": 100}})
    assert isinstance(comp, MyStore)
    assert comp.custom_param == 100


@pytest.mark.unit
def test_default_store_from_dict_without_type():
    with pytest.raises(DocumentStoreDeserializationError, match="Missing 'type' in DocumentStore serialization data"):
        default_document_store_from_dict(Mock, {})


@pytest.mark.unit
def test_default_store_from_dict_unregistered_store(request):
    # We use the test function name as store name to make sure it's not registered.
    # Since the registry is global we risk to have a store with the same name registered in another test.
    store_name = request.node.name

    with pytest.raises(
        DocumentStoreDeserializationError, match=f"DocumentStore '{store_name}' can't be deserialized as 'Mock'"
    ):
        default_document_store_from_dict(Mock, {"type": store_name})
