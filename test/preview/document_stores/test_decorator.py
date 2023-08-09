from unittest.mock import Mock

import pytest

from haystack.preview.testing.factory import store_class
from haystack.preview.document_stores.decorator import _default_store_to_dict, _default_store_from_dict
from haystack.preview.document_stores.errors import StoreDeserializationError


@pytest.mark.unit
def test_default_store_to_dict():
    MyStore = store_class("MyStore")
    comp = MyStore()
    res = _default_store_to_dict(comp)
    assert res == {"hash": id(comp), "type": "MyStore", "init_parameters": {}}


@pytest.mark.unit
def test_default_store_from_dict():
    MyStore = store_class("MyStore")
    comp = _default_store_from_dict(MyStore, {"type": "MyStore"})
    assert isinstance(comp, MyStore)


@pytest.mark.unit
def test_default_store_from_dict_without_type():
    with pytest.raises(StoreDeserializationError, match="Missing 'type' in store serialization data"):
        _default_store_from_dict(Mock, {})


@pytest.mark.unit
def test_default_store_from_dict_unregistered_store(request):
    # We use the test function name as store name to make sure it's not registered.
    # Since the registry is global we risk to have a store with the same name registered in another test.
    store_name = request.node.name

    with pytest.raises(StoreDeserializationError, match=f"Store '{store_name}' can't be deserialized as 'Mock'"):
        _default_store_from_dict(Mock, {"type": store_name})
