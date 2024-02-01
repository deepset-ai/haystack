import pytest

from haystack.core.component.types import Variadic
from haystack.core.errors import PipelineConnectError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.pipeline import parse_connect_string
from haystack.testing import factory


def test_parse_connection():
    assert parse_connect_string("foobar") == ("foobar", None)
    assert parse_connect_string("foo.bar") == ("foo", "bar")
    assert parse_connect_string("foo.inputs.baz") == ("foo", "baz")


class TestWithStrings:
    def test_connect(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.outputs.value", "comp2.inputs.value")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_reconnection(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.outputs.value", "comp2.inputs.value")
        pipe.connect("comp1.outputs.value", "comp2.inputs.value")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_sender_component_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2.inputs.value")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_receiver_component_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.outputs.value", "comp2")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_sender_and_receiver_component_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_sender_output_socket(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect(comp1.outputs.value, "comp2")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_receiver_input_socket(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.outputs.value", comp2.inputs.value)

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_sender_not_in_pipeline(self):
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp2", comp2)
        with pytest.raises(ValueError):
            pipe.connect("comp1.outputs.value", "comp2.inputs.value")

    def test_with_receiver_not_in_pipeline(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError):
            pipe.connect("comp1.outputs.value", "comp2.inputs.value")

    def test_with_sender_socket_name_not_in_pipeline(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.outputs.non_existing", "comp2.inputs.value")

    def test_with_receiver_socket_name_not_in_pipeline(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.outputs.value", "comp2.inputs.non_existing")

    def test_with_no_matching_types_and_same_names(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": str})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_with_multiple_sender_connections_with_same_type_and_differing_name(self):
        comp1 = factory.component_class("Comp1", output_types={"val1": int, "val2": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_with_multiple_receiver_connections_with_same_type_and_differing_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"val1": int, "val2": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_with_multiple_sender_connections_with_same_type_and_same_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int, "other": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_multiple_receiver_connections_with_same_type_and_same_name(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int, "other": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_multiple_outputs_to_non_variadic_input(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", output_types={"value": int})()
        comp3 = factory.component_class("Comp3", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.add_component("comp3", comp3)
        pipe.connect("comp1.outputs.value", "comp3.inputs.value")
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp2.outputs.value", "comp3.inputs.value")

    def test_multiple_outputs_to_variadic_input(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", output_types={"value": int})()
        comp3 = factory.component_class("Comp3", input_types={"value": Variadic[int]})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.add_component("comp3", comp3)
        pipe.connect("comp1.outputs.value", "comp3.inputs.value")
        pipe.connect("comp2.outputs.value", "comp3.inputs.value")

        assert comp1.outputs.value.receivers == ["comp3"]
        assert comp2.outputs.value.receivers == ["comp3"]
        assert comp3.inputs.value.senders == ["comp1", "comp2"]
        assert list(pipe.graph.edges) == [("comp1", "comp3", "value/value"), ("comp2", "comp3", "value/value")]


class TestWithSockets:
    def test_connect(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect(comp1.outputs.value, comp2.inputs.value)

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_sender_not_in_pipeline(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp2", comp2)
        with pytest.raises(ValueError):
            pipe.connect(comp1.outputs.value, comp2.inputs.value)

    def test_with_receiver_not_in_pipeline(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError):
            pipe.connect(comp1.outputs.value, comp2.inputs.value)

    def test_with_non_compatible_types(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": str})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect(comp1.outputs.value, comp2.inputs.value)

    def test_reconnection(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect(comp1.outputs.value, comp2.inputs.value)
        pipe.connect(comp1.outputs.value, comp2.inputs.value)

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_multiple_outputs_to_non_variadic_input(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", output_types={"value": int})()
        comp3 = factory.component_class("Comp3", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.add_component("comp3", comp3)
        pipe.connect(comp1.outputs.value, comp3.inputs.value)
        with pytest.raises(PipelineConnectError):
            pipe.connect(comp2.outputs.value, comp3.inputs.value)

    def test_multiple_outputs_to_variadic_input(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", output_types={"value": int})()
        comp3 = factory.component_class("Comp3", input_types={"value": Variadic[int]})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.add_component("comp3", comp3)
        pipe.connect(comp1.outputs.value, comp3.inputs.value)
        pipe.connect(comp2.outputs.value, comp3.inputs.value)

        assert comp1.outputs.value.receivers == ["comp3"]
        assert comp2.outputs.value.receivers == ["comp3"]
        assert comp3.inputs.value.senders == ["comp1", "comp2"]
        assert list(pipe.graph.edges) == [("comp1", "comp3", "value/value"), ("comp2", "comp3", "value/value")]

    def test_with_sender_as_string(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.outputs.value", comp2.inputs.value)

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_with_receiver_as_string(self):
        comp1 = factory.component_class("Comp1", output_types={"value": int})()
        comp2 = factory.component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect(comp1.outputs.value, "comp2.inputs.value")

        assert comp1.outputs.value.receivers == ["comp2"]
        assert comp2.inputs.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]
