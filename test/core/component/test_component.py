import logging
from typing import Any

import pytest

from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.component.types import Variadic
from haystack.core.errors import ComponentError
from haystack.core.pipeline import Pipeline


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
    with pytest.raises(ComponentError, match=r"must have a 'run\(\)' method"):

        @component
        class MockComponent:
            def another_method(self, input_value: int):
                return {"output_value": input_value}


def test_set_input_types():
    @component
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
    assert comp.__haystack_input__._sockets_dict == {"value": InputSocket("value", Any)}
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
    assert comp.__haystack_output__._sockets_dict == {"value": OutputSocket("value", int)}


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
    assert comp.__haystack_output__._sockets_dict == {"value": OutputSocket("value", int)}


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


def test_input_has_default_value():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int = 42):
            return {"value": value}

    comp = MockComponent()
    assert comp.__haystack_input__._sockets_dict["value"].default_value == 42
    assert not comp.__haystack_input__._sockets_dict["value"].is_mandatory


def test_keyword_only_args():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, *, arg: int):
            return {"value": arg}

    comp = MockComponent()
    component_inputs = {name: {"type": socket.type} for name, socket in comp.__haystack_input__._sockets_dict.items()}
    assert component_inputs == {"arg": {"type": int}}


def test_repr():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, value: int):
            return {"value": value}

    comp = MockComponent()
    assert repr(comp) == f"{object.__repr__(comp)}\nInputs:\n  - value: int\nOutputs:\n  - value: int"


def test_repr_added_to_pipeline():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, value: int):
            return {"value": value}

    pipe = Pipeline()
    comp = MockComponent()
    pipe.add_component("my_component", comp)
    assert repr(comp) == f"{object.__repr__(comp)}\nmy_component\nInputs:\n  - value: int\nOutputs:\n  - value: int"


def test_is_greedy_default_with_variadic_input():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: Variadic[int]):
            return {"value": value}

    assert not MockComponent.__haystack_is_greedy__
    assert not MockComponent().__haystack_is_greedy__


def test_is_greedy_default_without_variadic_input():
    @component
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    assert not MockComponent.__haystack_is_greedy__
    assert not MockComponent().__haystack_is_greedy__


def test_is_greedy_flag_with_variadic_input():
    @component(is_greedy=True)
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: Variadic[int]):
            return {"value": value}

    assert MockComponent.__haystack_is_greedy__
    assert MockComponent().__haystack_is_greedy__


def test_is_greedy_flag_without_variadic_input(caplog):
    caplog.set_level(logging.WARNING)

    @component(is_greedy=True)
    class MockComponent:
        @component.output_types(value=int)
        def run(self, value: int):
            return {"value": value}

    assert MockComponent.__haystack_is_greedy__
    assert caplog.text == ""
    assert MockComponent().__haystack_is_greedy__
    assert (
        "Component 'MockComponent' has no variadic input, but it's marked as greedy."
        " This is not supported and can lead to unexpected behavior.\n" in caplog.text
    )
