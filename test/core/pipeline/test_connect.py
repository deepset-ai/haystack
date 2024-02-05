import pytest

from haystack.core.component.types import Variadic
from haystack.core.errors import PipelineConnectError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.pipeline import parse_connect_string
from haystack.testing import factory


def test_parse_connection():
    assert parse_connect_string("foobar") == ("foobar", None)
    assert parse_connect_string("foo.bar") == ("foo", "bar")


def test_connect():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1.value", "comp2.value")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_reconnection():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1.value", "comp2.value")
    pipe.connect("comp1.value", "comp2.value")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_with_sender_component_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2.value")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_with_receiver_component_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1.value", "comp2")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_with_sender_and_receiver_component_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_with_sender_not_in_pipeline():
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp2", comp2)
    with pytest.raises(ValueError):
        pipe.connect("comp1.value", "comp2.value")


def test_with_receiver_not_in_pipeline():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    with pytest.raises(ValueError):
        pipe.connect("comp1.value", "comp2.value")


def test_with_sender_socket_name_not_in_pipeline():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp1.non_existing", "comp2.value")


def test_with_receiver_socket_name_not_in_pipeline():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp1.value", "comp2.non_existing")


def test_with_no_matching_types_and_same_names():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": str})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp1", "comp2")


def test_with_multiple_sender_connections_with_same_type_and_differing_name():
    comp1 = factory.component_class("Comp1", output_types={"val1": int, "val2": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp1", "comp2")


def test_with_multiple_receiver_connections_with_same_type_and_differing_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"val1": int, "val2": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp1", "comp2")


def test_with_multiple_sender_connections_with_same_type_and_same_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int, "other": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_with_multiple_receiver_connections_with_same_type_and_same_name():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", input_types={"value": int, "other": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2")

    assert comp1.__haystack_output__.value.receivers == ["comp2"]
    assert comp2.__haystack_input__.value.senders == ["comp1"]
    assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]


def test_multiple_outputs_to_non_variadic_input():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", output_types={"value": int})()
    comp3 = factory.component_class("Comp3", input_types={"value": int})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.add_component("comp3", comp3)
    pipe.connect("comp1.value", "comp3.value")
    with pytest.raises(PipelineConnectError):
        pipe.connect("comp2.value", "comp3.value")


def test_multiple_outputs_to_variadic_input():
    comp1 = factory.component_class("Comp1", output_types={"value": int})()
    comp2 = factory.component_class("Comp2", output_types={"value": int})()
    comp3 = factory.component_class("Comp3", input_types={"value": Variadic[int]})()
    pipe = Pipeline()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.add_component("comp3", comp3)
    pipe.connect("comp1.value", "comp3.value")
    pipe.connect("comp2.value", "comp3.value")

    assert comp1.__haystack_output__.value.receivers == ["comp3"]
    assert comp2.__haystack_output__.value.receivers == ["comp3"]
    assert comp3.__haystack_input__.value.senders == ["comp1", "comp2"]
    assert list(pipe.graph.edges) == [("comp1", "comp3", "value/value"), ("comp2", "comp3", "value/value")]
