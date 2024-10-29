# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import List, Optional
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
from haystack.core.component.types import InputSocket, OutputSocket, Variadic, GreedyVariadic, _empty
from haystack.core.errors import DeserializationError, PipelineConnectError, PipelineDrawingError, PipelineError
from haystack.core.pipeline import Pipeline, PredefinedPipeline
from haystack.core.pipeline.base import (
    _add_missing_input_defaults,
    _enqueue_component,
    _dequeue_component,
    _enqueue_waiting_component,
    _dequeue_waiting_component,
    _is_lazy_variadic,
)
from haystack.core.serialization import DeserializationCallbacks
from haystack.testing.factory import component_class
from haystack.testing.sample_components import AddFixedValue, Double, Greet

logging.basicConfig(level=logging.DEBUG)


@component
class FakeComponent:
    def __init__(self, an_init_param: Optional[str] = None):
        pass

    @component.output_types(value=str)
    def run(self, input_: str):
        return {"value": input_}


@component
class FakeComponentSquared:
    def __init__(self, an_init_param: Optional[str] = None):
        self.an_init_param = an_init_param
        self.inner = FakeComponent()

    @component.output_types(value=str)
    def run(self, input_: str):
        return {"value": input_}


class TestPipeline:
    """
    This class contains only unit tests for the Pipeline class.
    It doesn't test Pipeline.run(), that is done separately in a different way.
    """

    def test_pipeline_dumps(self, test_files_path):
        pipeline = Pipeline(max_runs_per_component=99)
        pipeline.add_component("Comp1", FakeComponent("Foo"))
        pipeline.add_component("Comp2", FakeComponent())
        pipeline.connect("Comp1.value", "Comp2.input_")
        result = pipeline.dumps()
        with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
            assert f.read() == result

    def test_pipeline_loads_invalid_data(self):
        invalid_yaml = """components:
        Comp1:
            init_parameters:
            an_init_param: null
            type: test.core.pipeline.test_pipeline.FakeComponent
        Comp2*
            init_parameters:
            an_init_param: null
            type: test.core.pipeline.test_pipeline.FakeComponent
        connections:
        * receiver: Comp2.input_
        sender: Comp1.value
        metadata:
        """

        with pytest.raises(DeserializationError, match="unmarshalling serialized"):
            pipeline = Pipeline.loads(invalid_yaml)

        invalid_init_parameter_yaml = """components:
        Comp1:
            init_parameters:
            unknown: null
            type: test.core.pipeline.test_pipeline.FakeComponent
        Comp2:
            init_parameters:
            an_init_param: null
            type: test.core.pipeline.test_pipeline.FakeComponent
        connections:
        - receiver: Comp2.input_
        sender: Comp1.value
        metadata: {}
        """

        with pytest.raises(DeserializationError, match=".*Comp1.*unknown.*"):
            pipeline = Pipeline.loads(invalid_init_parameter_yaml)

    def test_pipeline_dump(self, test_files_path, tmp_path):
        pipeline = Pipeline(max_runs_per_component=99)
        pipeline.add_component("Comp1", FakeComponent("Foo"))
        pipeline.add_component("Comp2", FakeComponent())
        pipeline.connect("Comp1.value", "Comp2.input_")
        with open(tmp_path / "out.yaml", "w") as f:
            pipeline.dump(f)
        # re-open and ensure it's the same data as the test file
        with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as test_f, open(tmp_path / "out.yaml", "r") as f:
            assert f.read() == test_f.read()

    def test_pipeline_load(self, test_files_path):
        with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
            pipeline = Pipeline.load(f)
            assert pipeline._max_runs_per_component == 99
            assert isinstance(pipeline.get_component("Comp1"), FakeComponent)
            assert isinstance(pipeline.get_component("Comp2"), FakeComponent)

    @patch("haystack.core.pipeline.base._to_mermaid_image")
    @patch("haystack.core.pipeline.base.is_in_jupyter")
    @patch("IPython.display.Image")
    @patch("IPython.display.display")
    def test_show_in_notebook(
        self, mock_ipython_display, mock_ipython_image, mock_is_in_jupyter, mock_to_mermaid_image
    ):
        pipe = Pipeline()

        mock_to_mermaid_image.return_value = b"some_image_data"
        mock_is_in_jupyter.return_value = True

        pipe.show()
        mock_ipython_image.assert_called_once_with(b"some_image_data")
        mock_ipython_display.assert_called_once()

    @patch("haystack.core.pipeline.base.is_in_jupyter")
    def test_show_not_in_notebook(self, mock_is_in_jupyter):
        pipe = Pipeline()

        mock_is_in_jupyter.return_value = False

        with pytest.raises(PipelineDrawingError):
            pipe.show()

    @patch("haystack.core.pipeline.base._to_mermaid_image")
    def test_draw(self, mock_to_mermaid_image, tmp_path):
        pipe = Pipeline()
        mock_to_mermaid_image.return_value = b"some_image_data"

        image_path = tmp_path / "test.png"
        pipe.draw(path=image_path)
        assert image_path.read_bytes() == mock_to_mermaid_image.return_value

    # UNIT
    def test_add_component_to_different_pipelines(self):
        first_pipe = Pipeline()
        second_pipe = Pipeline()
        some_component = component_class("Some")()

        assert some_component.__haystack_added_to_pipeline__ is None
        first_pipe.add_component("some", some_component)
        assert some_component.__haystack_added_to_pipeline__ is first_pipe

        with pytest.raises(PipelineError):
            second_pipe.add_component("some", some_component)

    def test_remove_component_raises_if_invalid_component_name(self):
        pipe = Pipeline()
        component = component_class("Some")()

        pipe.add_component("1", component)

        with pytest.raises(ValueError):
            pipe.remove_component("2")

    def test_remove_component_removes_component_and_its_edges(self):
        pipe = Pipeline()
        component_1 = component_class("Type1")()
        component_2 = component_class("Type2")()
        component_3 = component_class("Type3")()
        component_4 = component_class("Type4")()

        pipe.add_component("1", component_1)
        pipe.add_component("2", component_2)
        pipe.add_component("3", component_3)
        pipe.add_component("4", component_4)

        pipe.connect("1", "2")
        pipe.connect("2", "3")
        pipe.connect("3", "4")

        pipe.remove_component("2")

        assert ["1", "3", "4"] == sorted(pipe.graph.nodes)
        assert [("3", "4")] == sorted([(u, v) for (u, v) in pipe.graph.edges()])

    def test_remove_component_allows_you_to_reuse_the_component(self):
        pipe = Pipeline()
        Some = component_class("Some", input_types={"in": int}, output_types={"out": int})

        pipe.add_component("component_1", Some())
        pipe.add_component("component_2", Some())
        pipe.add_component("component_3", Some())
        pipe.connect("component_1", "component_2")
        pipe.connect("component_2", "component_3")
        component_2 = pipe.remove_component("component_2")

        assert component_2.__haystack_added_to_pipeline__ is None
        assert component_2.__haystack_input__._sockets_dict == {"in": InputSocket(name="in", type=int, senders=[])}
        assert component_2.__haystack_output__._sockets_dict == {
            "out": OutputSocket(name="out", type=int, receivers=[])
        }

        pipe2 = Pipeline()
        pipe2.add_component("component_4", Some())
        pipe2.add_component("component_2", component_2)
        pipe2.add_component("component_5", Some())

        pipe2.connect("component_4", "component_2")
        pipe2.connect("component_2", "component_5")
        assert component_2.__haystack_added_to_pipeline__ is pipe2
        assert component_2.__haystack_input__._sockets_dict == {
            "in": InputSocket(name="in", type=int, senders=["component_4"])
        }
        assert component_2.__haystack_output__._sockets_dict == {
            "out": OutputSocket(name="out", type=int, receivers=["component_5"])
        }

        # instance = pipe2.get_component("some")
        # assert instance == component

    # UNIT
    def test_get_component_name(self):
        pipe = Pipeline()
        some_component = component_class("Some")()
        pipe.add_component("some", some_component)

        assert pipe.get_component_name(some_component) == "some"

    # UNIT
    def test_get_component_name_not_added_to_pipeline(self):
        pipe = Pipeline()
        some_component = component_class("Some")()

        assert pipe.get_component_name(some_component) == ""

    # UNIT
    def test_repr(self):
        pipe = Pipeline(metadata={"test": "test"})
        pipe.add_component("add_two", AddFixedValue(add=2))
        pipe.add_component("add_default", AddFixedValue())
        pipe.add_component("double", Double())
        pipe.connect("add_two", "double")
        pipe.connect("double", "add_default")

        expected_repr = (
            f"{object.__repr__(pipe)}\n"
            "🧱 Metadata\n"
            "  - test: test\n"
            "🚅 Components\n"
            "  - add_two: AddFixedValue\n"
            "  - add_default: AddFixedValue\n"
            "  - double: Double\n"
            "🛤️ Connections\n"
            "  - add_two.result -> double.value (int)\n"
            "  - double.value -> add_default.value (int)\n"
        )

        assert repr(pipe) == expected_repr

    # UNIT
    def test_to_dict(self):
        add_two = AddFixedValue(add=2)
        add_default = AddFixedValue()
        double = Double()
        pipe = Pipeline(metadata={"test": "test"}, max_runs_per_component=42)
        pipe.add_component("add_two", add_two)
        pipe.add_component("add_default", add_default)
        pipe.add_component("double", double)
        pipe.connect("add_two", "double")
        pipe.connect("double", "add_default")

        res = pipe.to_dict()
        expected = {
            "metadata": {"test": "test"},
            "max_runs_per_component": 42,
            "components": {
                "add_two": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 2},
                },
                "add_default": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 1},
                },
                "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
            },
            "connections": [
                {"sender": "add_two.result", "receiver": "double.value"},
                {"sender": "double.value", "receiver": "add_default.value"},
            ],
        }
        assert res == expected

    def test_from_dict(self):
        data = {
            "metadata": {"test": "test"},
            "max_runs_per_component": 101,
            "components": {
                "add_two": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 2},
                },
                "add_default": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 1},
                },
                "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
            },
            "connections": [
                {"sender": "add_two.result", "receiver": "double.value"},
                {"sender": "double.value", "receiver": "add_default.value"},
            ],
        }
        pipe = Pipeline.from_dict(data)

        assert pipe.metadata == {"test": "test"}
        assert pipe._max_runs_per_component == 101

        # Components
        assert len(pipe.graph.nodes) == 3
        ## add_two
        add_two = pipe.graph.nodes["add_two"]
        assert add_two["instance"].add == 2
        assert add_two["input_sockets"] == {
            "value": InputSocket(name="value", type=int),
            "add": InputSocket(name="add", type=Optional[int], default_value=None),
        }
        assert add_two["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=["double"])}
        assert add_two["visits"] == 0

        ## add_default
        add_default = pipe.graph.nodes["add_default"]
        assert add_default["instance"].add == 1
        assert add_default["input_sockets"] == {
            "value": InputSocket(name="value", type=int, senders=["double"]),
            "add": InputSocket(name="add", type=Optional[int], default_value=None),
        }
        assert add_default["output_sockets"] == {"result": OutputSocket(name="result", type=int)}
        assert add_default["visits"] == 0

        ## double
        double = pipe.graph.nodes["double"]
        assert double["instance"]
        assert double["input_sockets"] == {"value": InputSocket(name="value", type=int, senders=["add_two"])}
        assert double["output_sockets"] == {"value": OutputSocket(name="value", type=int, receivers=["add_default"])}
        assert double["visits"] == 0

        # Connections
        connections = list(pipe.graph.edges(data=True))
        assert len(connections) == 2
        assert connections[0] == (
            "add_two",
            "double",
            {
                "conn_type": "int",
                "from_socket": OutputSocket(name="result", type=int, receivers=["double"]),
                "to_socket": InputSocket(name="value", type=int, senders=["add_two"]),
                "mandatory": True,
            },
        )
        assert connections[1] == (
            "double",
            "add_default",
            {
                "conn_type": "int",
                "from_socket": OutputSocket(name="value", type=int, receivers=["add_default"]),
                "to_socket": InputSocket(name="value", type=int, senders=["double"]),
                "mandatory": True,
            },
        )

    # TODO: Remove this, this should be a component test.
    # The pipeline can't handle this in any case nor way.
    def test_from_dict_with_callbacks(self):
        data = {
            "metadata": {"test": "test"},
            "components": {
                "add_two": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 2},
                },
                "add_default": {
                    "type": "haystack.testing.sample_components.add_value.AddFixedValue",
                    "init_parameters": {"add": 1},
                },
                "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
                "greet": {
                    "type": "haystack.testing.sample_components.greet.Greet",
                    "init_parameters": {"message": "test"},
                },
            },
            "connections": [
                {"sender": "add_two.result", "receiver": "double.value"},
                {"sender": "double.value", "receiver": "add_default.value"},
            ],
        }

        components_seen_in_callback = []

        def component_pre_init_callback(name, component_cls, init_params):
            assert name in ["add_two", "add_default", "double", "greet"]
            assert component_cls in [AddFixedValue, Double, Greet]

            if name == "add_two":
                assert init_params == {"add": 2}
            elif name == "add_default":
                assert init_params == {"add": 1}
            elif name == "greet":
                assert init_params == {"message": "test"}

            components_seen_in_callback.append(name)

        pipe = Pipeline.from_dict(
            data, callbacks=DeserializationCallbacks(component_pre_init=component_pre_init_callback)
        )
        assert components_seen_in_callback == ["add_two", "add_default", "double", "greet"]
        add_two = pipe.graph.nodes["add_two"]["instance"]
        assert add_two.add == 2
        add_default = pipe.graph.nodes["add_default"]["instance"]
        assert add_default.add == 1
        greet = pipe.graph.nodes["greet"]["instance"]
        assert greet.message == "test"
        assert greet.log_level == "INFO"

        def component_pre_init_callback_modify(name, component_cls, init_params):
            assert name in ["add_two", "add_default", "double", "greet"]
            assert component_cls in [AddFixedValue, Double, Greet]

            if name == "add_two":
                init_params["add"] = 3
            elif name == "add_default":
                init_params["add"] = 0
            elif name == "greet":
                init_params["message"] = "modified test"
                init_params["log_level"] = "DEBUG"

        pipe = Pipeline.from_dict(
            data, callbacks=DeserializationCallbacks(component_pre_init=component_pre_init_callback_modify)
        )
        add_two = pipe.graph.nodes["add_two"]["instance"]
        assert add_two.add == 3
        add_default = pipe.graph.nodes["add_default"]["instance"]
        assert add_default.add == 0
        greet = pipe.graph.nodes["greet"]["instance"]
        assert greet.message == "modified test"
        assert greet.log_level == "DEBUG"

        # Test with a component that internally instantiates another component
        def component_pre_init_callback_check_class(name, component_cls, init_params):
            assert name == "fake_component_squared"
            assert component_cls == FakeComponentSquared

        pipe = Pipeline()
        pipe.add_component("fake_component_squared", FakeComponentSquared())
        pipe = Pipeline.from_dict(
            pipe.to_dict(),
            callbacks=DeserializationCallbacks(component_pre_init=component_pre_init_callback_check_class),
        )
        assert type(pipe.graph.nodes["fake_component_squared"]["instance"].inner) == FakeComponent

    # UNIT
    def test_from_dict_with_empty_dict(self):
        assert Pipeline() == Pipeline.from_dict({})

    # TODO: UNIT, consider deprecating this argument
    def test_from_dict_with_components_instances(self):
        add_two = AddFixedValue(add=2)
        add_default = AddFixedValue()
        components = {"add_two": add_two, "add_default": add_default}
        data = {
            "metadata": {"test": "test"},
            "components": {
                "add_two": {},
                "add_default": {},
                "double": {"type": "haystack.testing.sample_components.double.Double", "init_parameters": {}},
            },
            "connections": [
                {"sender": "add_two.result", "receiver": "double.value"},
                {"sender": "double.value", "receiver": "add_default.value"},
            ],
        }
        pipe = Pipeline.from_dict(data, components=components)
        assert pipe.metadata == {"test": "test"}

        # Components
        assert len(pipe.graph.nodes) == 3
        ## add_two
        add_two_data = pipe.graph.nodes["add_two"]
        assert add_two_data["instance"] is add_two
        assert add_two_data["instance"].add == 2
        assert add_two_data["input_sockets"] == {
            "value": InputSocket(name="value", type=int),
            "add": InputSocket(name="add", type=Optional[int], default_value=None),
        }
        assert add_two_data["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=["double"])}
        assert add_two_data["visits"] == 0

        ## add_default
        add_default_data = pipe.graph.nodes["add_default"]
        assert add_default_data["instance"] is add_default
        assert add_default_data["instance"].add == 1
        assert add_default_data["input_sockets"] == {
            "value": InputSocket(name="value", type=int, senders=["double"]),
            "add": InputSocket(name="add", type=Optional[int], default_value=None),
        }
        assert add_default_data["output_sockets"] == {"result": OutputSocket(name="result", type=int, receivers=[])}
        assert add_default_data["visits"] == 0

        ## double
        double = pipe.graph.nodes["double"]
        assert double["instance"]
        assert double["input_sockets"] == {"value": InputSocket(name="value", type=int, senders=["add_two"])}
        assert double["output_sockets"] == {"value": OutputSocket(name="value", type=int, receivers=["add_default"])}
        assert double["visits"] == 0

        # Connections
        connections = list(pipe.graph.edges(data=True))
        assert len(connections) == 2
        assert connections[0] == (
            "add_two",
            "double",
            {
                "conn_type": "int",
                "from_socket": OutputSocket(name="result", type=int, receivers=["double"]),
                "to_socket": InputSocket(name="value", type=int, senders=["add_two"]),
                "mandatory": True,
            },
        )
        assert connections[1] == (
            "double",
            "add_default",
            {
                "conn_type": "int",
                "from_socket": OutputSocket(name="value", type=int, receivers=["add_default"]),
                "to_socket": InputSocket(name="value", type=int, senders=["double"]),
                "mandatory": True,
            },
        )

    # UNIT
    def test_from_dict_without_component_type(self):
        data = {
            "metadata": {"test": "test"},
            "components": {"add_two": {"init_parameters": {"add": 2}}},
            "connections": [],
        }
        with pytest.raises(PipelineError) as err:
            Pipeline.from_dict(data)

        err.match("Missing 'type' in component 'add_two'")

    # UNIT
    def test_from_dict_without_registered_component_type(self, request):
        data = {
            "metadata": {"test": "test"},
            "components": {"add_two": {"type": "foo.bar.baz", "init_parameters": {"add": 2}}},
            "connections": [],
        }
        with pytest.raises(PipelineError) as err:
            Pipeline.from_dict(data)

        err.match(r"Component .+ not imported.")

    # UNIT
    def test_from_dict_without_connection_sender(self):
        data = {"metadata": {"test": "test"}, "components": {}, "connections": [{"receiver": "some.receiver"}]}
        with pytest.raises(PipelineError) as err:
            Pipeline.from_dict(data)

        err.match("Missing sender in connection: {'receiver': 'some.receiver'}")

    # UNIT
    def test_from_dict_without_connection_receiver(self):
        data = {"metadata": {"test": "test"}, "components": {}, "connections": [{"sender": "some.sender"}]}
        with pytest.raises(PipelineError) as err:
            Pipeline.from_dict(data)

        err.match("Missing receiver in connection: {'sender': 'some.sender'}")

    def test_describe_input_only_no_inputs_components(self):
        A = component_class("A", input_types={}, output={"x": 0})
        B = component_class("B", input_types={}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
        p = Pipeline()
        p.add_component("a", A())
        p.add_component("b", B())
        p.add_component("c", C())
        p.connect("a.x", "c.x")
        p.connect("b.y", "c.y")
        assert p.inputs() == {}
        assert p.inputs(include_components_with_connected_inputs=True) == {
            "c": {"x": {"type": int, "is_mandatory": True}, "y": {"type": int, "is_mandatory": True}}
        }

    def test_describe_input_some_components_with_no_inputs(self):
        A = component_class("A", input_types={}, output={"x": 0})
        B = component_class("B", input_types={"y": int}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
        p = Pipeline()
        p.add_component("a", A())
        p.add_component("b", B())
        p.add_component("c", C())
        p.connect("a.x", "c.x")
        p.connect("b.y", "c.y")
        assert p.inputs() == {"b": {"y": {"type": int, "is_mandatory": True}}}
        assert p.inputs(include_components_with_connected_inputs=True) == {
            "b": {"y": {"type": int, "is_mandatory": True}},
            "c": {"x": {"type": int, "is_mandatory": True}, "y": {"type": int, "is_mandatory": True}},
        }

    def test_describe_input_all_components_have_inputs(self):
        A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
        B = component_class("B", input_types={"y": int}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
        p = Pipeline()
        p.add_component("a", A())
        p.add_component("b", B())
        p.add_component("c", C())
        p.connect("a.x", "c.x")
        p.connect("b.y", "c.y")
        assert p.inputs() == {
            "a": {"x": {"type": Optional[int], "is_mandatory": True}},
            "b": {"y": {"type": int, "is_mandatory": True}},
        }
        assert p.inputs(include_components_with_connected_inputs=True) == {
            "a": {"x": {"type": Optional[int], "is_mandatory": True}},
            "b": {"y": {"type": int, "is_mandatory": True}},
            "c": {"x": {"type": int, "is_mandatory": True}, "y": {"type": int, "is_mandatory": True}},
        }

    def test_describe_output_multiple_possible(self):
        """
        This pipeline has two outputs:
        {"b": {"output_b": {"type": str}}, "a": {"output_a": {"type": str}}}
        """
        A = component_class("A", input_types={"input_a": str}, output={"output_a": "str", "output_b": "str"})
        B = component_class("B", input_types={"input_b": str}, output={"output_b": "str"})

        pipe = Pipeline()
        pipe.add_component("a", A())
        pipe.add_component("b", B())
        pipe.connect("a.output_b", "b.input_b")

        assert pipe.outputs() == {"b": {"output_b": {"type": str}}, "a": {"output_a": {"type": str}}}
        assert pipe.outputs(include_components_with_connected_outputs=True) == {
            "a": {"output_a": {"type": str}, "output_b": {"type": str}},
            "b": {"output_b": {"type": str}},
        }

    def test_describe_output_single(self):
        """
        This pipeline has one output:
        {"c": {"z": {"type": int}}}
        """
        A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
        B = component_class("B", input_types={"y": int}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
        p = Pipeline()
        p.add_component("a", A())
        p.add_component("b", B())
        p.add_component("c", C())
        p.connect("a.x", "c.x")
        p.connect("b.y", "c.y")

        assert p.outputs() == {"c": {"z": {"type": int}}}
        assert p.outputs(include_components_with_connected_outputs=True) == {
            "a": {"x": {"type": int}},
            "b": {"y": {"type": int}},
            "c": {"z": {"type": int}},
        }

    def test_describe_no_outputs(self):
        """
        This pipeline sets up elaborate connections between three components but in fact it has no outputs:
        Check that p.outputs() == {}
        """
        A = component_class("A", input_types={"x": Optional[int]}, output={"x": 0})
        B = component_class("B", input_types={"y": int}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={})
        p = Pipeline()
        p.add_component("a", A())
        p.add_component("b", B())
        p.add_component("c", C())
        p.connect("a.x", "c.x")
        p.connect("b.y", "c.y")
        assert p.outputs() == {}
        assert p.outputs(include_components_with_connected_outputs=True) == {
            "a": {"x": {"type": int}},
            "b": {"y": {"type": int}},
        }

    def test_from_template(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
        pipe = Pipeline.from_template(PredefinedPipeline.INDEXING)
        assert pipe.get_component("cleaner")

    def test_walk_pipeline_with_no_cycles(self):
        """
        This pipeline has two source nodes, source1 and source2, one hello3 node in between, and one sink node, joiner.
        pipeline.walk() should return each component exactly once. The order is not guaranteed.
        """

        @component
        class Hello:
            @component.output_types(output=str)
            def run(self, word: str):
                """
                Takes a string in input and returns "Hello, <string>!" in output.
                """
                return {"output": f"Hello, {word}!"}

        @component
        class Joiner:
            @component.output_types(output=str)
            def run(self, word1: str, word2: str):
                """
                Takes two strings in input and returns "Hello, <string1> and <string2>!" in output.
                """
                return {"output": f"Hello, {word1} and {word2}!"}

        pipeline = Pipeline()
        source1 = Hello()
        source2 = Hello()
        hello3 = Hello()
        joiner = Joiner()
        pipeline.add_component("source1", source1)
        pipeline.add_component("source2", source2)
        pipeline.add_component("hello3", hello3)
        pipeline.add_component("joiner", joiner)

        pipeline.connect("source1", "joiner.word1")
        pipeline.connect("source2", "hello3")
        pipeline.connect("hello3", "joiner.word2")

        expected_components = [("source1", source1), ("source2", source2), ("joiner", joiner), ("hello3", hello3)]
        assert sorted(expected_components) == sorted(pipeline.walk())

    def test_walk_pipeline_with_cycles(self):
        """
        This pipeline consists of two components, which would run three times in a loop.
        pipeline.walk() should return these components exactly once. The order is not guaranteed.
        """

        @component
        class Hello:
            def __init__(self):
                self.iteration_counter = 0

            @component.output_types(intermediate=str, final=str)
            def run(self, word: str, intermediate: Optional[str] = None):
                """
                Takes a string in input and returns "Hello, <string>!" in output.
                """
                if self.iteration_counter < 3:
                    self.iteration_counter += 1
                    return {"intermediate": f"Hello, {intermediate or word}!"}
                return {"final": f"Hello, {intermediate or word}!"}

        pipeline = Pipeline()
        hello = Hello()
        hello_again = Hello()
        pipeline.add_component("hello", hello)
        pipeline.add_component("hello_again", hello_again)
        pipeline.connect("hello.intermediate", "hello_again.intermediate")
        pipeline.connect("hello_again.intermediate", "hello.intermediate")
        assert {("hello", hello), ("hello_again", hello_again)} == set(pipeline.walk())

    def test__init_graph(self):
        pipe = Pipeline()
        pipe.add_component("greet", Greet())
        pipe.add_component("adder", AddFixedValue())
        pipe.connect("greet", "adder")
        pipe._init_graph()
        for node in pipe.graph.nodes:
            assert pipe.graph.nodes[node]["visits"] == 0

    def test__normalize_varidiac_input_data(self):
        pipe = Pipeline()
        template = """
        Answer the following questions:
        {{ questions | join("\n") }}
        """
        pipe.add_component("prompt_builder", PromptBuilder(template=template))
        pipe.add_component("branch_joiner", BranchJoiner(type_=int))
        questions = ["What is the capital of Italy?", "What is the capital of France?"]
        data = {
            "prompt_builder": {"questions": questions},
            "branch_joiner": {"value": 1},
            "not_a_component": "some input data",
        }
        res = pipe._normalize_varidiac_input_data(data)
        assert res == {
            "prompt_builder": {"questions": ["What is the capital of Italy?", "What is the capital of France?"]},
            "branch_joiner": {"value": [1]},
            "not_a_component": "some input data",
        }

    def test__prepare_component_input_data(self):
        MockComponent = component_class("MockComponent", input_types={"x": List[str], "y": str})
        pipe = Pipeline()
        pipe.add_component("first_mock", MockComponent())
        pipe.add_component("second_mock", MockComponent())

        res = pipe._prepare_component_input_data({"x": ["some data"], "y": "some other data"})
        assert res == {
            "first_mock": {"x": ["some data"], "y": "some other data"},
            "second_mock": {"x": ["some data"], "y": "some other data"},
        }
        assert id(res["first_mock"]["x"]) != id(res["second_mock"]["x"])

    def test__prepare_component_input_data_with_connected_inputs(self):
        MockComponent = component_class(
            "MockComponent", input_types={"x": List[str], "y": str}, output_types={"z": str}
        )
        pipe = Pipeline()
        pipe.add_component("first_mock", MockComponent())
        pipe.add_component("second_mock", MockComponent())
        pipe.connect("first_mock.z", "second_mock.y")

        res = pipe._prepare_component_input_data({"x": ["some data"], "y": "some other data"})
        assert res == {"first_mock": {"x": ["some data"], "y": "some other data"}, "second_mock": {"x": ["some data"]}}
        assert id(res["first_mock"]["x"]) != id(res["second_mock"]["x"])

    def test__prepare_component_input_data_with_non_existing_input(self, caplog):
        pipe = Pipeline()
        res = pipe._prepare_component_input_data({"input_name": 1})
        assert res == {}
        assert (
            "Inputs ['input_name'] were not matched to any component inputs, "
            "please check your run parameters." in caplog.text
        )

    def test_connect(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        assert pipe.connect("comp1.value", "comp2.value") is pipe

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_already_connected(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.value", "comp2.value")
        pipe.connect("comp1.value", "comp2.value")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_sender_component_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2.value")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_receiver_component_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.value", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_sender_and_receiver_component_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_sender_not_in_pipeline(self):
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp2", comp2)
        with pytest.raises(ValueError):
            pipe.connect("comp1.value", "comp2.value")

    def test_connect_with_receiver_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError):
            pipe.connect("comp1.value", "comp2.value")

    def test_connect_with_sender_socket_name_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.non_existing", "comp2.value")

    def test_connect_with_receiver_socket_name_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.value", "comp2.non_existing")

    def test_connect_with_no_matching_types_and_same_names(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": str})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_sender_connections_with_same_type_and_differing_name(self):
        comp1 = component_class("Comp1", output_types={"val1": int, "val2": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_receiver_connections_with_same_type_and_differing_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"val1": int, "val2": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_sender_connections_with_same_type_and_same_name(self):
        comp1 = component_class("Comp1", output_types={"value": int, "other": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_multiple_receiver_connections_with_same_type_and_same_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int, "other": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_multiple_outputs_to_non_variadic_input(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", output_types={"value": int})()
        comp3 = component_class("Comp3", input_types={"value": int})()
        pipe = Pipeline()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.add_component("comp3", comp3)
        pipe.connect("comp1.value", "comp3.value")
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp2.value", "comp3.value")

    def test_connect_multiple_outputs_to_variadic_input(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", output_types={"value": int})()
        comp3 = component_class("Comp3", input_types={"value": Variadic[int]})()
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

    def test_connect_same_component_as_sender_and_receiver(self):
        """
        This pipeline consists of one component, which would be connected to itself.
        Connecting a component to itself is raises PipelineConnectError.
        """
        pipe = Pipeline()
        single_component = FakeComponent()
        pipe.add_component("single_component", single_component)
        with pytest.raises(PipelineConnectError):
            pipe.connect("single_component.out", "single_component.in")

    def test__run_component(self, spying_tracer, caplog):
        caplog.set_level(logging.INFO)
        sentence_builder = component_class(
            "SentenceBuilder", input_types={"words": List[str]}, output={"text": "some words"}
        )()
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output={"doc": Document(content="some words")}
        )()
        document_cleaner = component_class(
            "DocumentCleaner",
            input_types={"doc": Document},
            output={"cleaned_doc": Document(content="some cleaner words")},
        )()

        pipe = Pipeline()
        pipe.add_component("sentence_builder", sentence_builder)
        pipe.add_component("document_builder", document_builder)
        pipe.add_component("document_cleaner", document_cleaner)
        pipe.connect("sentence_builder.text", "document_builder.text")
        pipe.connect("document_builder.doc", "document_cleaner.doc")
        assert spying_tracer.spans == []
        res = pipe._run_component("document_builder", {"text": "whatever"})
        assert res == {"doc": Document(content="some words")}

        assert len(spying_tracer.spans) == 1
        span = spying_tracer.spans[0]
        assert span.operation_name == "haystack.component.run"
        assert span.tags == {
            "haystack.component.name": "document_builder",
            "haystack.component.type": "DocumentBuilder",
            "haystack.component.input_types": {"text": "str"},
            "haystack.component.input_spec": {"text": {"type": "str", "senders": ["sentence_builder"]}},
            "haystack.component.output_spec": {"doc": {"type": "Document", "receivers": ["document_cleaner"]}},
            "haystack.component.visits": 1,
        }

        assert caplog.messages == ["Running component document_builder"]

    def test__run_component_with_variadic_input(self):
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        pipe = Pipeline()
        pipe.add_component("document_joiner", document_joiner)
        inputs = {"docs": [Document(content="doc1"), Document(content="doc2")]}
        pipe._run_component("document_joiner", inputs)
        assert inputs == {"docs": []}

    def test__component_has_enough_inputs_to_run(self):
        sentence_builder = component_class("SentenceBuilder", input_types={"words": List[str]})()
        pipe = Pipeline()
        pipe.add_component("sentence_builder", sentence_builder)

        assert not pipe._component_has_enough_inputs_to_run("sentence_builder", {})
        assert not pipe._component_has_enough_inputs_to_run(
            "sentence_builder", {"sentence_builder": {"wrong_input_name": "blah blah"}}
        )
        assert pipe._component_has_enough_inputs_to_run(
            "sentence_builder", {"sentence_builder": {"words": ["blah blah"]}}
        )

    def test__find_components_that_will_receive_no_input(self):
        sentence_builder = component_class(
            "SentenceBuilder", input_types={"words": List[str]}, output_types={"text": str}
        )()
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        conditional_document_builder = component_class(
            "ConditionalDocumentBuilder", output_types={"doc": Document, "noop": None}
        )()

        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        pipe = Pipeline()
        pipe.add_component("sentence_builder", sentence_builder)
        pipe.add_component("document_builder", document_builder)
        pipe.add_component("document_joiner", document_joiner)
        pipe.add_component("conditional_document_builder", conditional_document_builder)
        pipe.connect("sentence_builder.text", "document_builder.text")
        pipe.connect("document_builder.doc", "document_joiner.docs")
        pipe.connect("conditional_document_builder.doc", "document_joiner.docs")

        res = pipe._find_components_that_will_receive_no_input("sentence_builder", {}, {})
        assert res == {("document_builder", document_builder), ("document_joiner", document_joiner)}

        res = pipe._find_components_that_will_receive_no_input("sentence_builder", {"text": "some text"}, {})
        assert res == set()

        res = pipe._find_components_that_will_receive_no_input("conditional_document_builder", {"noop": None}, {})
        assert res == {("document_joiner", document_joiner)}

        res = pipe._find_components_that_will_receive_no_input(
            "conditional_document_builder", {"noop": None}, {"document_joiner": {"docs": []}}
        )
        assert res == {("document_joiner", document_joiner)}

        res = pipe._find_components_that_will_receive_no_input(
            "conditional_document_builder", {"noop": None}, {"document_joiner": {"docs": [Document("some text")]}}
        )
        assert res == set()

        multiple_outputs = component_class("MultipleOutputs", output_types={"first": int, "second": int})()

        def custom_init(self):
            component.set_input_type(self, "first", Optional[int], 1)
            component.set_input_type(self, "second", Optional[int], 2)

        multiple_optional_inputs = component_class("MultipleOptionalInputs", extra_fields={"__init__": custom_init})()

        pipe = Pipeline()
        pipe.add_component("multiple_outputs", multiple_outputs)
        pipe.add_component("multiple_optional_inputs", multiple_optional_inputs)
        pipe.connect("multiple_outputs.second", "multiple_optional_inputs.first")

        res = pipe._find_components_that_will_receive_no_input("multiple_outputs", {"first": 1}, {})
        assert res == {("multiple_optional_inputs", multiple_optional_inputs)}

        res = pipe._find_components_that_will_receive_no_input(
            "multiple_outputs", {"first": 1}, {"multiple_optional_inputs": {"second": 200}}
        )
        assert res == set()

        res = pipe._find_components_that_will_receive_no_input("multiple_outputs", {"second": 1}, {})
        assert res == set()

    def test__distribute_output(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document, "another_doc": Document}
        )()
        document_cleaner = component_class(
            "DocumentCleaner", input_types={"doc": Document}, output_types={"cleaned_doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        pipe = Pipeline()
        pipe.add_component("document_builder", document_builder)
        pipe.add_component("document_cleaner", document_cleaner)
        pipe.add_component("document_joiner", document_joiner)
        pipe.connect("document_builder.doc", "document_cleaner.doc")
        pipe.connect("document_builder.another_doc", "document_joiner.docs")

        inputs = {"document_builder": {"text": "some text"}}
        run_queue = []
        waiting_queue = [("document_joiner", document_joiner)]
        receivers = [
            (
                "document_cleaner",
                OutputSocket("doc", Document, ["document_cleaner"]),
                InputSocket("doc", Document, _empty, ["document_builder"]),
            ),
            (
                "document_joiner",
                OutputSocket("another_doc", Document, ["document_joiner"]),
                InputSocket("docs", Variadic[Document], _empty, ["document_builder"]),
            ),
        ]
        res = pipe._distribute_output(
            receivers, {"doc": Document("some text"), "another_doc": Document()}, inputs, run_queue, waiting_queue
        )

        assert res == {}
        assert inputs == {
            "document_builder": {"text": "some text"},
            "document_cleaner": {"doc": Document("some text")},
            "document_joiner": {"docs": [Document()]},
        }
        assert run_queue == [("document_cleaner", document_cleaner)]
        assert waiting_queue == [("document_joiner", document_joiner)]

    def test__find_next_runnable_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        pipe = Pipeline()
        components_inputs = {"document_builder": {"text": "some text"}}
        waiting_queue = [("document_builder", document_builder)]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)
        assert pair == ("document_builder", document_builder)

    def test__find_next_runnable_component_without_component_inputs(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        pipe = Pipeline()
        components_inputs = {}
        waiting_queue = [("document_builder", document_builder)]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)
        assert pair == ("document_builder", document_builder)

    def test__find_next_runnable_component_with_component_with_only_variadic_non_greedy_input(self):
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        pipe = Pipeline()
        components_inputs = {}
        waiting_queue = [("document_joiner", document_joiner)]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)
        assert pair == ("document_joiner", document_joiner)

    def test__find_next_runnable_component_with_component_with_only_default_input(self):
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")

        pipe = Pipeline()
        components_inputs = {}
        waiting_queue = [("prompt_builder", prompt_builder)]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)

        assert pair == ("prompt_builder", prompt_builder)

    def test__find_next_runnable_component_with_component_with_variadic_non_greedy_and_default_input(self):
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")

        pipe = Pipeline()
        components_inputs = {}
        waiting_queue = [("prompt_builder", prompt_builder), ("document_joiner", document_joiner)]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)

        assert pair == ("document_joiner", document_joiner)

    def test__find_next_runnable_component_with_different_components_inputs(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")

        pipe = Pipeline()
        components_inputs = {"document_builder": {"text": "some text"}}
        waiting_queue = [
            ("prompt_builder", prompt_builder),
            ("document_builder", document_builder),
            ("document_joiner", document_joiner),
        ]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)

        assert pair == ("document_builder", document_builder)

    def test__find_next_runnable_component_with_different_components_without_any_input(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")

        pipe = Pipeline()
        components_inputs = {}
        waiting_queue = [
            ("prompt_builder", prompt_builder),
            ("document_builder", document_builder),
            ("document_joiner", document_joiner),
        ]
        pair = pipe._find_next_runnable_component(components_inputs, waiting_queue)

        assert pair == ("document_builder", document_builder)

    def test__is_stuck_in_a_loop(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")

        pipe = Pipeline()

        waiting_queue = [("document_builder", document_builder)]
        assert pipe._is_stuck_in_a_loop(waiting_queue)

        waiting_queue = [("document_joiner", document_joiner)]
        assert pipe._is_stuck_in_a_loop(waiting_queue)

        waiting_queue = [("prompt_builder", prompt_builder)]
        assert pipe._is_stuck_in_a_loop(waiting_queue)

        waiting_queue = [("document_joiner", document_joiner), ("prompt_builder", prompt_builder)]
        assert not pipe._is_stuck_in_a_loop(waiting_queue)

        waiting_queue = [("document_builder", document_joiner), ("prompt_builder", prompt_builder)]
        assert not pipe._is_stuck_in_a_loop(waiting_queue)

        waiting_queue = [("document_builder", document_joiner), ("document_joiner", document_joiner)]
        assert not pipe._is_stuck_in_a_loop(waiting_queue)

    def test__enqueue_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        run_queue = []
        waiting_queue = []
        _enqueue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == [("document_builder", document_builder)]
        assert waiting_queue == []

        run_queue = [("document_builder", document_builder)]
        waiting_queue = []
        _enqueue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == [("document_builder", document_builder)]
        assert waiting_queue == []

        run_queue = []
        waiting_queue = [("document_builder", document_builder)]
        _enqueue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == [("document_builder", document_builder)]
        assert waiting_queue == []

        run_queue = []
        waiting_queue = [("document_joiner", document_joiner)]
        _enqueue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == [("document_builder", document_builder)]
        assert waiting_queue == [("document_joiner", document_joiner)]

        run_queue = [("document_joiner", document_joiner)]
        waiting_queue = []
        _enqueue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == [("document_joiner", document_joiner), ("document_builder", document_builder)]
        assert waiting_queue == []

    def test__dequeue_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        run_queue = []
        waiting_queue = []
        _dequeue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == []
        assert waiting_queue == []

        run_queue = [("document_builder", document_builder)]
        waiting_queue = []
        _dequeue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == []
        assert waiting_queue == []

        run_queue = []
        waiting_queue = [("document_builder", document_builder)]
        _dequeue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == []
        assert waiting_queue == []

        run_queue = [("document_builder", document_builder)]
        waiting_queue = [("document_builder", document_builder)]
        _dequeue_component(("document_builder", document_builder), run_queue, waiting_queue)
        assert run_queue == []
        assert waiting_queue == []

        run_queue = [("document_builder", document_builder)]
        waiting_queue = [("document_builder", document_builder)]
        _dequeue_component(("document_joiner", document_joiner), run_queue, waiting_queue)
        assert run_queue == [("document_builder", document_builder)]
        assert waiting_queue == [("document_builder", document_builder)]

    def test__add_missing_input_defaults(self):
        name = "prompt_builder"
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")
        components_inputs = {}
        _add_missing_input_defaults(name, prompt_builder, components_inputs)
        assert components_inputs == {"prompt_builder": {"questions": "", "template": None, "template_variables": None}}

        name = "answer_builder"
        answer_builder = AnswerBuilder()
        components_inputs = {"answer_builder": {"query": "What is the answer?"}}
        _add_missing_input_defaults(name, answer_builder, components_inputs)
        assert components_inputs == {
            "answer_builder": {
                "query": "What is the answer?",
                "meta": None,
                "documents": None,
                "pattern": None,
                "reference_pattern": None,
            }
        }

        name = "branch_joiner"
        branch_joiner = BranchJoiner(int)
        components_inputs = {}
        _add_missing_input_defaults(name, branch_joiner, components_inputs)
        assert components_inputs == {"branch_joiner": {}}

    def test__find_next_runnable_lazy_variadic_or_default_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()
        prompt_builder = PromptBuilder(template="{{ questions | join('\n') }}")
        pipe = Pipeline()

        waiting_queue = [("document_builder", document_builder)]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("document_builder", document_builder)

        waiting_queue = [("document_joiner", document_joiner)]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("document_joiner", document_joiner)

        waiting_queue = [("prompt_builder", prompt_builder)]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("prompt_builder", prompt_builder)

        waiting_queue = [
            ("document_builder", document_builder),
            ("document_joiner", document_joiner),
            ("prompt_builder", prompt_builder),
        ]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("document_joiner", document_joiner)

        waiting_queue = [
            ("prompt_builder", prompt_builder),
            ("document_builder", document_builder),
            ("document_joiner", document_joiner),
        ]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("prompt_builder", prompt_builder)

        waiting_queue = [
            ("document_builder", document_builder),
            ("document_joiner", document_joiner),
            ("prompt_builder", prompt_builder),
        ]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("document_joiner", document_joiner)

        waiting_queue = [
            ("document_builder", document_builder),
            ("prompt_builder", prompt_builder),
            ("document_joiner", document_joiner),
        ]
        pair = pipe._find_next_runnable_lazy_variadic_or_default_component(waiting_queue)
        assert pair == ("prompt_builder", prompt_builder)

    def test__enqueue_waiting_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        waiting_queue = []
        _enqueue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_builder", document_builder)]

        waiting_queue = [("document_builder", document_builder)]
        _enqueue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_builder", document_builder)]

        waiting_queue = [("document_joiner", document_joiner)]
        _enqueue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_joiner", document_joiner), ("document_builder", document_builder)]

        waiting_queue = [("document_builder", document_builder), ("document_joiner", document_joiner)]
        _enqueue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_builder", document_builder), ("document_joiner", document_joiner)]

    def test__dequeue_waiting_component(self):
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        waiting_queue = []
        _dequeue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == []

        waiting_queue = [("document_builder", document_builder)]
        _dequeue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == []

        waiting_queue = [("document_joiner", document_joiner)]
        _dequeue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_joiner", document_joiner)]

        waiting_queue = [("document_builder", document_builder), ("document_joiner", document_joiner)]
        _dequeue_waiting_component(("document_builder", document_builder), waiting_queue)
        assert waiting_queue == [("document_joiner", document_joiner)]

    def test__is_lazy_variadic(self):
        VariadicAndGreedyVariadic = component_class(
            "VariadicAndGreedyVariadic", input_types={"variadic": Variadic[int], "greedy_variadic": GreedyVariadic[int]}
        )
        NonVariadic = component_class("NonVariadic", input_types={"value": int})
        VariadicNonGreedyVariadic = component_class(
            "VariadicNonGreedyVariadic", input_types={"variadic": Variadic[int]}
        )
        NonVariadicAndGreedyVariadic = component_class(
            "NonVariadicAndGreedyVariadic", input_types={"greedy_variadic": GreedyVariadic[int]}
        )
        assert not _is_lazy_variadic(VariadicAndGreedyVariadic())
        assert not _is_lazy_variadic(NonVariadic())
        assert _is_lazy_variadic(VariadicNonGreedyVariadic())
        assert not _is_lazy_variadic(NonVariadicAndGreedyVariadic())

    def test__find_receivers_from(self):
        sentence_builder = component_class(
            "SentenceBuilder", input_types={"words": List[str]}, output_types={"text": str}
        )()
        document_builder = component_class(
            "DocumentBuilder", input_types={"text": str}, output_types={"doc": Document}
        )()
        conditional_document_builder = component_class(
            "ConditionalDocumentBuilder", output_types={"doc": Document, "noop": None}
        )()

        document_joiner = component_class("DocumentJoiner", input_types={"docs": Variadic[Document]})()

        pipe = Pipeline()
        pipe.add_component("sentence_builder", sentence_builder)
        pipe.add_component("document_builder", document_builder)
        pipe.add_component("document_joiner", document_joiner)
        pipe.add_component("conditional_document_builder", conditional_document_builder)
        pipe.connect("sentence_builder.text", "document_builder.text")
        pipe.connect("document_builder.doc", "document_joiner.docs")
        pipe.connect("conditional_document_builder.doc", "document_joiner.docs")

        res = pipe._find_receivers_from("sentence_builder")
        assert res == [
            (
                "document_builder",
                OutputSocket(name="text", type=str, receivers=["document_builder"]),
                InputSocket(name="text", type=str, default_value=_empty, senders=["sentence_builder"]),
            )
        ]

        res = pipe._find_receivers_from("document_builder")
        assert res == [
            (
                "document_joiner",
                OutputSocket(name="doc", type=Document, receivers=["document_joiner"]),
                InputSocket(
                    name="docs",
                    type=Variadic[Document],
                    default_value=_empty,
                    senders=["document_builder", "conditional_document_builder"],
                ),
            )
        ]

        res = pipe._find_receivers_from("document_joiner")
        assert res == []

        res = pipe._find_receivers_from("conditional_document_builder")
        assert res == [
            (
                "document_joiner",
                OutputSocket(name="doc", type=Document, receivers=["document_joiner"]),
                InputSocket(
                    name="docs",
                    type=Variadic[Document],
                    default_value=_empty,
                    senders=["document_builder", "conditional_document_builder"],
                ),
            )
        ]
