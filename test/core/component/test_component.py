from typing import Any

import pytest

from haystack.core.component import Component, InputSocket, OutputSocket, component
from haystack.core.component.component import _InputOutput
from haystack.core.errors import ComponentError
from haystack.core.pipeline import Pipeline
from haystack.testing.factory import component_class


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
    assert comp.__haystack_input__ == {"value": InputSocket("value", Any)}
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
    assert comp.__haystack_output__ == {"value": OutputSocket("value", int)}


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
    assert comp.__haystack_output__ == {"value": OutputSocket("value", int)}


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
    assert comp.__haystack_input__["value"].default_value == 42
    assert not comp.__haystack_input__["value"].is_mandatory


def test_keyword_only_args():
    @component
    class MockComponent:
        def __init__(self):
            component.set_output_types(self, value=int)

        def run(self, *, arg: int):
            return {"value": arg}

    comp = MockComponent()
    component_inputs = {name: {"type": socket.type} for name, socket in comp.__haystack_input__.items()}
    assert component_inputs == {"arg": {"type": int}}


def test_component_with_inputs_field():
    @component
    class MockComponent:
        inputs = []

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    with pytest.raises(ComponentError):
        MockComponent()


def test_component_with_outputs_field():
    @component
    class MockComponent:
        outputs = []

        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @component.output_types(output_value=int)
        def run(self, input_value: int):
            return {"output_value": input_value}

    with pytest.raises(ComponentError):
        MockComponent()


class TestInputOutput:
    def test_init(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_input__)
        assert io._component == comp
        assert io._sockets == comp.__haystack_input__
        assert "input_1" in io.__dict__
        assert io.__dict__["input_1"] == comp.__haystack_input__["input_1"]
        assert "input_2" in io.__dict__
        assert io.__dict__["input_2"] == comp.__haystack_input__["input_2"]

    def test_init_without_sockets(self):
        comp = component_class("SomeComponent")()
        with pytest.raises(ValueError):
            _InputOutput(component=comp, sockets=None)

    def test_init_with_mixed_sockets(self):
        comp = component_class("SomeComponent", input_types={"input_1": int}, output_types={"output_1": int})()
        sockets = {**comp.__haystack_input__, **comp.__haystack_output__}
        with pytest.raises(ValueError):
            _InputOutput(component=comp, sockets=sockets)

    def test_init_without_component(self):
        with pytest.raises(ValueError):
            _InputOutput(component=None, sockets={})

    def test_init_with_empty_sockets(self):
        comp = component_class("SomeComponent")()
        io = _InputOutput(component=comp, sockets={})

        assert io._component == comp
        assert io._sockets == {}

    def test_component_name(self):
        comp = component_class("SomeComponent")()
        io = _InputOutput(component=comp, sockets={})
        assert io._component_name() == "SomeComponent"

    def test_component_name_added_to_pipeline(self):
        comp = component_class("SomeComponent")()
        pipeline = Pipeline()
        pipeline.add_component("my_component", comp)

        io = _InputOutput(component=comp, sockets={})
        assert io._component_name() == "my_component"

    def test_socket_repr_input(self):
        comp = component_class("SomeComponent", input_types={"input_1": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_input__)

        assert io._socket_repr("input_1") == "SomeComponent.inputs.input_1"

        pipeline = Pipeline()
        pipeline.add_component("my_component", comp)

        assert io._socket_repr("input_1") == "my_component.inputs.input_1"

    def test_socket_repr_output(self):
        comp = component_class("SomeComponent", output_types={"output_1": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_output__)

        assert io._socket_repr("output_1") == "SomeComponent.outputs.output_1"

        pipeline = Pipeline()
        pipeline.add_component("my_component", comp)

        assert io._socket_repr("output_1") == "my_component.outputs.output_1"

    def test_getattribute(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_input__)

        assert io.input_1 == "SomeComponent.inputs.input_1"
        assert io.input_2 == "SomeComponent.inputs.input_2"

        pipeline = Pipeline()
        pipeline.add_component("my_component", comp)

        assert io.input_1 == "my_component.inputs.input_1"
        assert io.input_2 == "my_component.inputs.input_2"

    def test_getattribute_non_existing_socket(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_input__)

        with pytest.raises(AttributeError):
            io.input_3

    def test_repr(self):
        comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
        io = _InputOutput(component=comp, sockets=comp.__haystack_input__)
        res = repr(io)
        assert res == "SomeComponent inputs:\n  - input_1: int\n  - input_2: int"
