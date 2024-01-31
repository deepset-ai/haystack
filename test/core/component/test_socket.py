import pytest

from haystack.core.component.component import _InputOutput
from haystack.core.pipeline import Pipeline
from haystack.testing.factory import component_class


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
