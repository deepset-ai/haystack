from typing import Any

import pytest

from canals import component
from canals.errors import ComponentError


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
