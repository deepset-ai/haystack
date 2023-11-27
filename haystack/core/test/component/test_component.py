import typing
from typing import Any, Optional

import pytest

from canals import component
from canals.component.descriptions import find_component_inputs, find_component_outputs
from canals.errors import ComponentError
from canals.component import InputSocket, OutputSocket, Component


def test_correct_declaration():
    @component
    class MockComponent:
        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies also instantiation works with no issues
    assert MockComponent()
    assert component.registry["test_component.MockComponent"] == MockComponent


def test_correct_declaration_with_additional_readonly_property():
    @component
    class MockComponent:
        @property
        def store(self):
            return "test_store"

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert MockComponent()
    assert component.registry["test_component.MockComponent"] == MockComponent
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

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    # Verifies that instantiation works with no issues
    assert component.registry["test_component.MockComponent"] == MockComponent
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

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(value=int)
        def run(self, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert comp.__canals_input__ == {"value": InputSocket("value", Any)}
    assert comp.run() == {"value": 1}


def test_set_output_types():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert comp.__canals_output__ == {"value": OutputSocket("value", int)}


def test_output_types_decorator_with_compatible_type():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert comp.__canals_output__ == {"value": OutputSocket("value", int)}


def test_component_decorator_set_it_as_component():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": 1}

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    comp = MockComponent()
    assert isinstance(comp, Component)


def test_inputs_method_no_inputs():
    @component
    class MockComponent:
        def run(self):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {}


def test_inputs_method_one_input():
    @component
    class MockComponent:
        def run(self, value: int):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {"value": {"is_mandatory": True, "is_variadic": False, "type": int}}


def test_inputs_method_multiple_inputs():
    @component
    class MockComponent:
        def run(self, value1: int, value2: str):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {
        "value1": {"is_mandatory": True, "is_variadic": False, "type": int},
        "value2": {"is_mandatory": True, "is_variadic": False, "type": str},
    }


def test_inputs_method_multiple_inputs_optional():
    @component
    class MockComponent:
        def run(self, value1: int, value2: Optional[str]):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {
        "value1": {"is_mandatory": True, "is_variadic": False, "type": int},
        "value2": {"is_mandatory": True, "is_variadic": False, "type": typing.Optional[str]},
    }


def test_inputs_method_variadic_positional_args():
    @component
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=Any)

        def run(self, *args):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {"value": {"is_mandatory": True, "is_variadic": False, "type": typing.Any}}


def test_inputs_method_variadic_keyword_positional_args():
    @component
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=Any)

        def run(self, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {"value": {"is_mandatory": True, "is_variadic": False, "type": typing.Any}}


def test_inputs_dynamic_from_init():
    @component
    class MockComponent:
        def __init__(self):
            component.set_input_types(self, value=int)

        def run(self, value: int, **kwargs):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_inputs(comp) == {"value": {"is_mandatory": True, "is_variadic": False, "type": int}}


def test_outputs_method_no_outputs():
    @component
    class MockComponent:
        def run(self):
            return {}

    comp = MockComponent()
    assert find_component_outputs(comp) == {}


def test_outputs_method_one_output():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_outputs(comp) == {"value": {"type": int}}


def test_outputs_method_multiple_outputs():
    @component
    class MockComponent:
        @component.output_types(value1=int, value2=str)
        def run(self):
            return {"value1": 1, "value2": "test"}

    comp = MockComponent()
    assert find_component_outputs(comp) == {"value1": {"type": int}, "value2": {"type": str}}


def test_outputs_dynamic_from_init():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self):
            return {"value": 1}

    comp = MockComponent()
    assert find_component_outputs(comp) == {"value": {"type": int}}
