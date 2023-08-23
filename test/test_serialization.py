from unittest.mock import Mock

import pytest

from canals.errors import DeserializationError
from canals.testing import factory
from canals.serialization import default_to_dict, default_from_dict


def test_default_component_to_dict():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = default_to_dict(comp)
    assert res == {
        "type": "MyComponent",
        "init_parameters": {},
    }


def test_default_component_to_dict_with_init_parameters():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = default_to_dict(comp, some_key="some_value")
    assert res == {
        "type": "MyComponent",
        "init_parameters": {"some_key": "some_value"},
    }


def test_default_component_from_dict():
    def custom_init(self, some_param):
        self.some_param = some_param

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    comp = default_from_dict(
        MyComponent,
        {
            "type": "MyComponent",
            "init_parameters": {
                "some_param": 10,
            },
        },
    )
    assert isinstance(comp, MyComponent)
    assert comp.some_param == 10


def test_default_component_from_dict_without_type():
    with pytest.raises(DeserializationError, match="Missing 'type' in serialization data"):
        default_from_dict(Mock, {})


def test_default_component_from_dict_unregistered_component(request):
    # We use the test function name as component name to make sure it's not registered.
    # Since the registry is global we risk to have a component with the same name registered in another test.
    component_name = request.node.name

    with pytest.raises(DeserializationError, match=f"Class '{component_name}' can't be deserialized as 'Mock'"):
        default_from_dict(Mock, {"type": component_name})
