from typing import Any
from unittest.mock import Mock

import pytest

from canals import component
from canals.component.component import _default_component_to_dict, _default_component_from_dict
from canals.testing import factory
from canals.errors import ComponentError, ComponentDeserializationError


def test_correct_declaration():
    @component
    class MockComponent:
        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent


def test_correct_declaration_with_additional_readonly_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert MockComponent()
    assert component.registry["MockComponent"] == MockComponent
    assert MockComponent().store == "test_store"


def test_correct_declaration_with_additional_writable_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        @store.setter
        def store(self, value):
            self._store = value

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert component.registry["MockComponent"] == MockComponent
    comp = MockComponent()
    comp.store = "test_store"
    assert comp.store == "test_store"


def test_missing_run():
    with pytest.raises(ComponentError, match="must have a 'run\(\)' method"):

        @component
        class MockComponent:
            def another_method(self, input_value: int):
                return {"output_value": input_value}


def test_set_input_types():
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=Any)

        @component.output_types(value=int)
        def run(self, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert comp.run.__canals_input__ == {
        "value": {
            "name": "value",
            "type": Any,
            "is_optional": False,
        }
    }
    assert comp.run() == {"value": 1}


def test_set_output_types():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.run.__canals_output__ == {
        "value": {
            "name": "value",
            "type": int,
        }
    }


def test_output_types_decorator_with_compatible_type():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.run.__canals_output__ == {
        "value": {
            "name": "value",
            "type": int,
        }
    }


def test_component_decorator_set_it_as_component():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.__canals_component__


def test_default_component_to_dict():
    MyComponent = factory.component_class("MyComponent")
    comp = MyComponent()
    res = _default_component_to_dict(comp)
    assert res == {
        "hash": id(comp),
        "type": "MyComponent",
        "init_parameters": {},
    }


def test_default_component_to_dict_with_init_parameters():
    extra_fields = {"init_parameters": {"some_key": "some_value"}}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    comp = MyComponent()
    res = _default_component_to_dict(comp)
    assert res == {
        "hash": id(comp),
        "type": "MyComponent",
        "init_parameters": {"some_key": "some_value"},
    }


def test_default_component_from_dict():
    def custom_init(self, some_param):
        self.some_param = some_param

    extra_fields = {"__init__": custom_init}
    MyComponent = factory.component_class("MyComponent", extra_fields=extra_fields)
    comp = _default_component_from_dict(
        MyComponent,
        {
            "type": "MyComponent",
            "init_parameters": {
                "some_param": 10,
            },
            "hash": 1234,
        },
    )
    assert isinstance(comp, MyComponent)
    assert comp.some_param == 10


def test_default_component_from_dict_without_type():
    with pytest.raises(ComponentDeserializationError, match="Missing 'type' in component serialization data"):
        _default_component_from_dict(Mock, {})


def test_default_component_from_dict_unregistered_component(request):
    # We use the test function name as component name to make sure it's not registered.
    # Since the registry is global we risk to have a component with the same name registered in another test.
    component_name = request.node.name

    with pytest.raises(
        ComponentDeserializationError, match=f"Component '{component_name}' can't be deserialized as 'Mock'"
    ):
        _default_component_from_dict(Mock, {"type": component_name})
