import pytest

from haystack.core.component._input_output import InputOutput
from haystack.core.pipeline import Pipeline
from haystack.testing.factory import component_class


def test_init():
    comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_input__)
    assert io._component == comp
    assert io._sockets == comp.__haystack_input__
    assert "input_1" in io.__dict__
    assert io.__dict__["input_1"] == comp.__haystack_input__["input_1"]
    assert "input_2" in io.__dict__
    assert io.__dict__["input_2"] == comp.__haystack_input__["input_2"]


def test_init_without_sockets():
    comp = component_class("SomeComponent")()
    with pytest.raises(ValueError):
        InputOutput(component=comp, sockets=None)


def test_init_with_mixed_sockets():
    comp = component_class("SomeComponent", input_types={"input_1": int}, output_types={"output_1": int})()
    sockets = {**comp.__haystack_input__, **comp.__haystack_output__}
    with pytest.raises(ValueError):
        InputOutput(component=comp, sockets=sockets)


def test_init_without_component():
    with pytest.raises(ValueError):
        InputOutput(component=None, sockets={})


def test_init_with_empty_sockets():
    comp = component_class("SomeComponent")()
    io = InputOutput(component=comp, sockets={})

    assert io._component == comp
    assert io._sockets == {}


def test_component_name():
    comp = component_class("SomeComponent")()
    io = InputOutput(component=comp, sockets={})
    assert io._component_name() == "SomeComponent"


def test_component_name_added_to_pipeline():
    comp = component_class("SomeComponent")()
    pipeline = Pipeline()
    pipeline.add_component("my_component", comp)

    io = InputOutput(component=comp, sockets={})
    assert io._component_name() == "my_component"


def test_socket_repr_input():
    comp = component_class("SomeComponent", input_types={"input_1": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_input__)

    assert io._socket_repr("input_1") == "SomeComponent.inputs.input_1"

    pipeline = Pipeline()
    pipeline.add_component("my_component", comp)

    assert io._socket_repr("input_1") == "my_component.inputs.input_1"


def test_socket_repr_output():
    comp = component_class("SomeComponent", output_types={"output_1": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_output__)

    assert io._socket_repr("output_1") == "SomeComponent.outputs.output_1"

    pipeline = Pipeline()
    pipeline.add_component("my_component", comp)

    assert io._socket_repr("output_1") == "my_component.outputs.output_1"


def test_getattribute():
    comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_input__)

    assert io.input_1 == "SomeComponent.inputs.input_1"
    assert io.input_2 == "SomeComponent.inputs.input_2"

    pipeline = Pipeline()
    pipeline.add_component("my_component", comp)

    assert io.input_1 == "my_component.inputs.input_1"
    assert io.input_2 == "my_component.inputs.input_2"


def test_getattribute_non_existing_socket():
    comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_input__)

    with pytest.raises(AttributeError):
        io.input_3


def test_repr():
    comp = component_class("SomeComponent", input_types={"input_1": int, "input_2": int})()
    io = InputOutput(component=comp, sockets=comp.__haystack_input__)
    res = repr(io)
    assert res == "SomeComponent inputs:\n  - input_1: int\n  - input_2: int"
