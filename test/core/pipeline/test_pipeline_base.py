# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional
from unittest.mock import patch

import pytest

from pandas import DataFrame

from haystack import Document
from haystack.core.component import component
from haystack.core.component.types import InputSocket, OutputSocket, Variadic, GreedyVariadic, _empty
from haystack.core.errors import (
    DeserializationError,
    PipelineConnectError,
    PipelineDrawingError,
    PipelineError,
    PipelineMaxComponentRuns,
)
from haystack.core.pipeline import PredefinedPipeline
from haystack.core.pipeline.base import PipelineBase
from haystack.core.pipeline.base import ComponentPriority, _NO_OUTPUT_PRODUCED
from haystack.core.pipeline.utils import FIFOPriorityQueue

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


@pytest.fixture
def regular_output_socket():
    """Output socket for a regular (non-variadic) connection with receivers"""
    return OutputSocket("output1", int, receivers=["receiver1", "receiver2"])


@pytest.fixture
def regular_input_socket():
    """Regular (non-variadic) input socket with a single sender"""
    return InputSocket("input1", int, senders=["sender1"])


@pytest.fixture
def lazy_variadic_input_socket():
    """Lazy variadic input socket with multiple senders"""
    return InputSocket("variadic_input", Variadic[int], senders=["sender1", "sender2"])


class TestPipelineBase:
    """
    This class contains only unit tests for the PipelineBase class.
    It doesn't test Pipeline.run(), that is done separately in a different way.
    """

    def test_pipeline_dumps(self, test_files_path):
        pipeline = PipelineBase(max_runs_per_component=99)
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
            type: test.core.pipeline.test_pipeline_base.FakeComponent
        Comp2*
            init_parameters:
            an_init_param: null
            type: test.core.pipeline.test_pipeline_base.FakeComponent
        connections:
        * receiver: Comp2.input_
        sender: Comp1.value
        metadata:
        """

        with pytest.raises(DeserializationError, match="unmarshalling serialized"):
            pipeline = PipelineBase.loads(invalid_yaml)

        invalid_init_parameter_yaml = """components:
        Comp1:
            init_parameters:
            unknown: null
            type: test.core.pipeline.test_pipeline_base.FakeComponent
        Comp2:
            init_parameters:
            an_init_param: null
            type: test.core.pipeline.test_pipeline_base.FakeComponent
        connections:
        - receiver: Comp2.input_
        sender: Comp1.value
        metadata: {}
        """

        with pytest.raises(DeserializationError, match=".*Comp1.*unknown.*"):
            pipeline = PipelineBase.loads(invalid_init_parameter_yaml)

    def test_pipeline_dump(self, test_files_path, tmp_path):
        pipeline = PipelineBase(max_runs_per_component=99)
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
            pipeline = PipelineBase.load(f)
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
        pipe = PipelineBase()

        mock_to_mermaid_image.return_value = b"some_image_data"
        mock_is_in_jupyter.return_value = True

        pipe.show()
        mock_ipython_image.assert_called_once_with(b"some_image_data")
        mock_ipython_display.assert_called_once()

    @patch("haystack.core.pipeline.base.is_in_jupyter")
    def test_show_not_in_notebook(self, mock_is_in_jupyter):
        pipe = PipelineBase()

        mock_is_in_jupyter.return_value = False

        with pytest.raises(PipelineDrawingError):
            pipe.show()

    @patch("haystack.core.pipeline.base._to_mermaid_image")
    def test_draw(self, mock_to_mermaid_image, tmp_path):
        pipe = PipelineBase()
        mock_to_mermaid_image.return_value = b"some_image_data"

        image_path = tmp_path / "test.png"
        pipe.draw(path=image_path)
        assert image_path.read_bytes() == mock_to_mermaid_image.return_value

    def test_find_super_components(self):
        """
        Test that the pipeline can find super components in it's pipeline.
        """
        from haystack import Pipeline
        from haystack.components.converters import MultiFileConverter
        from haystack.components.preprocessors import DocumentPreprocessor
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        multi_file_converter = MultiFileConverter()
        doc_processor = DocumentPreprocessor()

        pipeline = Pipeline()
        pipeline.add_component("converter", multi_file_converter)
        pipeline.add_component("preprocessor", doc_processor)
        pipeline.add_component("writer", DocumentWriter(document_store=InMemoryDocumentStore()))
        pipeline.connect("converter", "preprocessor")
        pipeline.connect("preprocessor", "writer")

        result = pipeline._find_super_components()

        assert len(result) == 2
        assert [("converter", multi_file_converter), ("preprocessor", doc_processor)] == result

    def test_merge_super_component_pipelines(self):
        from haystack import Pipeline
        from haystack.components.converters import MultiFileConverter
        from haystack.components.preprocessors import DocumentPreprocessor
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        multi_file_converter = MultiFileConverter()
        doc_processor = DocumentPreprocessor()

        pipeline = Pipeline()
        pipeline.add_component("converter", multi_file_converter)
        pipeline.add_component("preprocessor", doc_processor)
        pipeline.add_component("writer", DocumentWriter(document_store=InMemoryDocumentStore()))
        pipeline.connect("converter", "preprocessor")
        pipeline.connect("preprocessor", "writer")

        merged_graph, super_component_components = pipeline._merge_super_component_pipelines()

        assert super_component_components == {
            "router": "converter",
            "docx": "converter",
            "html": "converter",
            "json": "converter",
            "md": "converter",
            "text": "converter",
            "pdf": "converter",
            "pptx": "converter",
            "xlsx": "converter",
            "joiner": "converter",
            "csv": "converter",
            "splitter": "preprocessor",
            "cleaner": "preprocessor",
        }

        expected_nodes = [
            "cleaner",
            "csv",
            "docx",
            "html",
            "joiner",
            "json",
            "md",
            "pdf",
            "pptx",
            "router",
            "splitter",
            "text",
            "writer",
            "xlsx",
        ]
        assert sorted(merged_graph.nodes) == expected_nodes

        expected_edges = [
            ("cleaner", "writer"),
            ("csv", "joiner"),
            ("docx", "joiner"),
            ("html", "joiner"),
            ("joiner", "splitter"),
            ("json", "joiner"),
            ("md", "joiner"),
            ("pdf", "joiner"),
            ("pptx", "joiner"),
            ("router", "csv"),
            ("router", "docx"),
            ("router", "html"),
            ("router", "json"),
            ("router", "md"),
            ("router", "pdf"),
            ("router", "pptx"),
            ("router", "text"),
            ("router", "xlsx"),
            ("splitter", "cleaner"),
            ("text", "joiner"),
            ("xlsx", "joiner"),
        ]
        actual_edges = [(u, v) for u, v, _ in merged_graph.edges]
        assert sorted(actual_edges) == expected_edges

    # UNIT
    def test_add_invalid_component_name(self):
        pipe = PipelineBase()
        with pytest.raises(ValueError):
            pipe.add_component("this.is.not.a.valida.name", FakeComponent)
        with pytest.raises(ValueError):
            pipe.add_component("_debug", FakeComponent)

    def test_add_component_to_different_pipelines(self):
        first_pipe = PipelineBase()
        second_pipe = PipelineBase()
        some_component = component_class("Some")()

        assert some_component.__haystack_added_to_pipeline__ is None
        first_pipe.add_component("some", some_component)
        assert some_component.__haystack_added_to_pipeline__ is first_pipe

        with pytest.raises(PipelineError):
            second_pipe.add_component("some", some_component)

    def test_remove_component_raises_if_invalid_component_name(self):
        pipe = PipelineBase()
        component = component_class("Some")()

        pipe.add_component("1", component)

        with pytest.raises(ValueError):
            pipe.remove_component("2")

    def test_remove_component_removes_component_and_its_edges(self):
        pipe = PipelineBase()
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
        pipe = PipelineBase()
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

        pipe2 = PipelineBase()
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
        pipe = PipelineBase()
        some_component = component_class("Some")()
        pipe.add_component("some", some_component)

        assert pipe.get_component_name(some_component) == "some"

    # UNIT
    def test_get_component_name_not_added_to_pipeline(self):
        pipe = PipelineBase()
        some_component = component_class("Some")()

        assert pipe.get_component_name(some_component) == ""

    # UNIT
    def test_repr(self):
        pipe = PipelineBase(metadata={"test": "test"})
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
        pipe = PipelineBase(metadata={"test": "test"}, max_runs_per_component=42)
        pipe.add_component("add_two", add_two)
        pipe.add_component("add_default", add_default)
        pipe.add_component("double", double)
        pipe.connect("add_two", "double")
        pipe.connect("double", "add_default")

        res = pipe.to_dict()
        expected = {
            "metadata": {"test": "test"},
            "max_runs_per_component": 42,
            "connection_type_validation": True,
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
        pipe = PipelineBase.from_dict(data)

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

        pipe = PipelineBase.from_dict(
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

        pipe = PipelineBase.from_dict(
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

        pipe = PipelineBase()
        pipe.add_component("fake_component_squared", FakeComponentSquared())
        pipe = PipelineBase.from_dict(
            pipe.to_dict(),
            callbacks=DeserializationCallbacks(component_pre_init=component_pre_init_callback_check_class),
        )
        assert type(pipe.graph.nodes["fake_component_squared"]["instance"].inner) == FakeComponent

    # UNIT
    def test_from_dict_with_empty_dict(self):
        assert PipelineBase() == PipelineBase.from_dict({})

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
        pipe = PipelineBase.from_dict(data, components=components)
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
            PipelineBase.from_dict(data)

        err.match("Missing 'type' in component 'add_two'")

    # UNIT
    def test_from_dict_without_registered_component_type(self):
        data = {
            "metadata": {"test": "test"},
            "components": {"add_two": {"type": "foo.bar.baz", "init_parameters": {"add": 2}}},
            "connections": [],
        }
        with pytest.raises(PipelineError) as err:
            PipelineBase.from_dict(data)

        err.match(r"Component .+ not imported.")

    def test_from_dict_with_invalid_type(self):
        data = {
            "metadata": {"test": "test"},
            "components": {"add_two": {"type": "", "init_parameters": {"add": 2}}},
            "connections": [],
        }
        with pytest.raises(PipelineError) as err:
            PipelineBase.from_dict(data)

        err.match(
            r"Component '' \(name: 'add_two'\) not imported. Please check that the package is installed and the component path is correct."
        )

    def test_from_dict_with_correct_import_but_invalid_type(self):
        # Test case: Module imports but component not found in registry.
        data_registry_error = {
            "metadata": {"test": "test"},
            "components": {"add_two": {"type": "haystack.testing.NonExistentComponent", "init_parameters": {"add": 2}}},
            "connections": [],
        }

        # Patch thread_safe_import so it doesn't raise an ImportError.
        with patch("haystack.utils.type_serialization.thread_safe_import") as mock_import:
            mock_import.return_value = None
            with pytest.raises(PipelineError) as err_info:
                PipelineBase.from_dict(data_registry_error)
            outer_message = str(err_info.value)
            inner_message = str(err_info.value.__cause__)

            assert "Component 'haystack.testing.NonExistentComponent' (name: 'add_two') not imported." in outer_message
            assert "Successfully imported module 'haystack.testing' but couldn't find" in inner_message
            assert "in the component registry." in inner_message
            assert "registered under a different path." in inner_message

    # UNIT
    def test_from_dict_without_connection_sender(self):
        data = {"metadata": {"test": "test"}, "components": {}, "connections": [{"receiver": "some.receiver"}]}
        with pytest.raises(PipelineError) as err:
            PipelineBase.from_dict(data)

        err.match("Missing sender in connection: {'receiver': 'some.receiver'}")

    # UNIT
    def test_from_dict_without_connection_receiver(self):
        data = {"metadata": {"test": "test"}, "components": {}, "connections": [{"sender": "some.sender"}]}
        with pytest.raises(PipelineError) as err:
            PipelineBase.from_dict(data)

        err.match("Missing receiver in connection: {'sender': 'some.sender'}")

    def test_describe_input_only_no_inputs_components(self):
        A = component_class("A", input_types={}, output={"x": 0})
        B = component_class("B", input_types={}, output={"y": 0})
        C = component_class("C", input_types={"x": int, "y": int}, output={"z": 0})
        p = PipelineBase()
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
        p = PipelineBase()
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
        p = PipelineBase()
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

        pipe = PipelineBase()
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
        p = PipelineBase()
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
        p = PipelineBase()
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
        pipe = PipelineBase.from_template(PredefinedPipeline.INDEXING)
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

        pipeline = PipelineBase()
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

        pipeline = PipelineBase()
        hello = Hello()
        hello_again = Hello()
        pipeline.add_component("hello", hello)
        pipeline.add_component("hello_again", hello_again)
        pipeline.connect("hello.intermediate", "hello_again.intermediate")
        pipeline.connect("hello_again.intermediate", "hello.intermediate")
        assert {("hello", hello), ("hello_again", hello_again)} == set(pipeline.walk())

    def test__prepare_component_input_data(self):
        MockComponent = component_class("MockComponent", input_types={"x": List[str], "y": str})
        pipe = PipelineBase()
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
        pipe = PipelineBase()
        pipe.add_component("first_mock", MockComponent())
        pipe.add_component("second_mock", MockComponent())
        pipe.connect("first_mock.z", "second_mock.y")

        res = pipe._prepare_component_input_data({"x": ["some data"], "y": "some other data"})
        assert res == {"first_mock": {"x": ["some data"], "y": "some other data"}, "second_mock": {"x": ["some data"]}}
        assert id(res["first_mock"]["x"]) != id(res["second_mock"]["x"])

    def test__prepare_component_input_data_with_non_existing_input(self, caplog):
        pipe = PipelineBase()
        res = pipe._prepare_component_input_data({"input_name": 1})
        assert res == {}
        assert (
            "Inputs ['input_name'] were not matched to any component inputs, "
            "please check your run parameters." in caplog.text
        )

    def test_connect(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        assert pipe.connect("comp1.value", "comp2.value") is pipe

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_already_connected(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
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
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2.value")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_receiver_component_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.value", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_sender_and_receiver_component_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_sender_not_in_pipeline(self):
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp2", comp2)
        with pytest.raises(ValueError):
            pipe.connect("comp1.value", "comp2.value")

    def test_connect_with_receiver_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError):
            pipe.connect("comp1.value", "comp2.value")

    def test_connect_with_sender_socket_name_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.non_existing", "comp2.value")

    def test_connect_with_receiver_socket_name_not_in_pipeline(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1.value", "comp2.non_existing")

    def test_connect_with_no_matching_types_and_same_names(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": str})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_sender_connections_with_same_type_and_differing_name(self):
        comp1 = component_class("Comp1", output_types={"val1": int, "val2": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_receiver_connections_with_same_type_and_differing_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"val1": int, "val2": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        with pytest.raises(PipelineConnectError):
            pipe.connect("comp1", "comp2")

    def test_connect_with_multiple_sender_connections_with_same_type_and_same_name(self):
        comp1 = component_class("Comp1", output_types={"value": int, "other": int})()
        comp2 = component_class("Comp2", input_types={"value": int})()
        pipe = PipelineBase()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1", "comp2")

        assert comp1.__haystack_output__.value.receivers == ["comp2"]
        assert comp2.__haystack_input__.value.senders == ["comp1"]
        assert list(pipe.graph.edges) == [("comp1", "comp2", "value/value")]

    def test_connect_with_multiple_receiver_connections_with_same_type_and_same_name(self):
        comp1 = component_class("Comp1", output_types={"value": int})()
        comp2 = component_class("Comp2", input_types={"value": int, "other": int})()
        pipe = PipelineBase()
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
        pipe = PipelineBase()
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
        pipe = PipelineBase()
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
        pipe = PipelineBase()
        single_component = FakeComponent()
        pipe.add_component("single_component", single_component)
        with pytest.raises(PipelineConnectError):
            pipe.connect("single_component.out", "single_component.in")

    @pytest.mark.parametrize(
        "component_inputs,sockets,expected_inputs",
        [
            ({"mandatory": 1}, {"mandatory": InputSocket("mandatory", int)}, {"mandatory": 1}),
            ({}, {"optional": InputSocket("optional", str, default_value="test")}, {"optional": "test"}),
            (
                {"mandatory": 1},
                {
                    "mandatory": InputSocket("mandatory", int),
                    "optional": InputSocket("optional", str, default_value="test"),
                },
                {"mandatory": 1, "optional": "test"},
            ),
            (
                {},
                {"optional_variadic": InputSocket("optional_variadic", Variadic[str], default_value="test")},
                {"optional_variadic": ["test"]},
            ),
            (
                {},
                {
                    "optional_1": InputSocket("optional_1", int, default_value=1),
                    "optional_2": InputSocket("optional_2", int, default_value=2),
                },
                {"optional_1": 1, "optional_2": 2},
            ),
        ],
        ids=["no-defaults", "only-default", "mixed-default", "variadic-default", "multiple_defaults"],
    )
    def test__add_missing_defaults(self, component_inputs, sockets, expected_inputs):
        filled_inputs = PipelineBase._add_missing_input_defaults(component_inputs, sockets)

        assert filled_inputs == expected_inputs

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

        pipe = PipelineBase()
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

    @pytest.mark.parametrize(
        "component, inputs, expected_priority, test_description",
        [
            # Test case 1: BLOCKED - Missing mandatory input
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "mandatory_input": InputSocket("mandatory_input", int),
                        "optional_input": InputSocket(
                            "optional_input", str, default_value="default", senders=["previous_component"]
                        ),
                    },
                },
                {"optional_input": [{"sender": "previous_component", "value": "test"}]},
                ComponentPriority.BLOCKED,
                "Component should be BLOCKED when mandatory input is missing",
            ),
            # Test case 2: BLOCKED - No trigger after first visit
            (
                {
                    "instance": "mock_instance",
                    "visits": 1,  # Already visited
                    "input_sockets": {
                        "mandatory_input": InputSocket("mandatory_input", int),
                        "optional_input": InputSocket("optional_input", str, default_value="default"),
                    },
                },
                {"mandatory_input": [{"sender": None, "value": 42}]},
                ComponentPriority.BLOCKED,
                "Component should be BLOCKED when there's no new trigger after first visit",
            ),
            # Test case 3: HIGHEST - Greedy socket ready
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "greedy_input": InputSocket("greedy_input", GreedyVariadic[int], senders=["component1"]),
                        "normal_input": InputSocket("normal_input", str, senders=["component2"]),
                    },
                },
                {
                    "greedy_input": [{"sender": "component1", "value": 42}],
                    "normal_input": [{"sender": "component2", "value": "test"}],
                },
                ComponentPriority.HIGHEST,
                "Component should have HIGHEST priority when greedy socket has valid input",
            ),
            # Test case 4: DEFER - Greedy socket ready but optional missing
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "greedy_input": InputSocket("greedy_input", GreedyVariadic[int], senders=["component1"]),
                        "optional_input": InputSocket(
                            "optional_input", str, senders=["component2"], default_value="test"
                        ),
                    },
                },
                {"greedy_input": [{"sender": "component1", "value": 42}]},
                ComponentPriority.DEFER,
                "Component should DEFER when greedy socket has valid input but expected optional input is missing",
            ),
            # Test case 4: READY - All predecessors executed
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "mandatory_input": InputSocket("mandatory_input", int, senders=["previous_component"]),
                        "optional_input": InputSocket(
                            "optional_input", str, senders=["another_component"], default_value="default"
                        ),
                    },
                },
                {
                    "mandatory_input": [{"sender": "previous_component", "value": 42}],
                    "optional_input": [{"sender": "another_component", "value": "test"}],
                },
                ComponentPriority.READY,
                "Component should be READY when all predecessors have executed",
            ),
            # Test case 5: DEFER - Lazy variadic sockets resolved and optional missing.
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "variadic_input": InputSocket(
                            "variadic_input", Variadic[int], senders=["component1", "component2"]
                        ),
                        "normal_input": InputSocket("normal_input", str, senders=["component3"]),
                        "optional_input": InputSocket(
                            "optional_input", str, default_value="default", senders=["component4"]
                        ),
                    },
                },
                {
                    "variadic_input": [
                        {"sender": "component1", "value": "test"},
                        {"sender": "component2", "value": _NO_OUTPUT_PRODUCED},
                    ],
                    "normal_input": [{"sender": "component3", "value": "test"}],
                },
                ComponentPriority.DEFER,
                "Component should DEFER when all lazy variadic sockets are resolved",
            ),
            # Test case 6: DEFER_LAST - Incomplete variadic inputs
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {
                        "variadic_input": InputSocket(
                            "variadic_input", Variadic[int], senders=["component1", "component2"]
                        ),
                        "normal_input": InputSocket("normal_input", str),
                    },
                },
                {
                    "variadic_input": [{"sender": "component1", "value": 42}],  # Missing component2
                    "normal_input": [{"sender": "component3", "value": "test"}],
                },
                ComponentPriority.DEFER_LAST,
                "Component should be DEFER_LAST when not all variadic senders have produced output",
            ),
            # Test case 7: READY - No input sockets, first visit
            (
                {
                    "instance": "mock_instance",
                    "visits": 0,
                    "input_sockets": {"optional_input": InputSocket("optional_input", str, default_value="default")},
                },
                {},  # no inputs
                ComponentPriority.READY,
                "Component should be READY on first visit when it has no input sockets",
            ),
            # Test case 8: BLOCKED - No connected input sockets, subsequent visit
            (
                {
                    "instance": "mock_instance",
                    "visits": 1,
                    "input_sockets": {"optional_input": InputSocket("optional_input", str, default_value="default")},
                },
                {},  # no inputs
                ComponentPriority.BLOCKED,
                "Component should be BLOCKED on subsequent visits when it has no input sockets",
            ),
        ],
        ids=lambda p: p.name if isinstance(p, ComponentPriority) else str(p),
    )
    def test__calculate_priority(self, component, inputs, expected_priority, test_description):
        """Test priority calculation for various component and input combinations."""
        # For variadic inputs, set up senders if needed
        for socket in component["input_sockets"].values():
            if socket.is_variadic and not hasattr(socket, "senders"):
                socket.senders = ["component1", "component2"]

        assert PipelineBase._calculate_priority(component, inputs) == expected_priority

    @pytest.mark.parametrize(
        "pipeline_inputs,expected_output",
        [
            # Test case 1: Empty input
            ({}, {}),
            # Test case 2: Single component, multiple inputs
            (
                {"component1": {"input1": 42, "input2": "test", "input3": True}},
                {
                    "component1": {
                        "input1": [{"sender": None, "value": 42}],
                        "input2": [{"sender": None, "value": "test"}],
                        "input3": [{"sender": None, "value": True}],
                    }
                },
            ),
            # Test case 3: Multiple components
            (
                {
                    "component1": {"input1": 42, "input2": "test"},
                    "component2": {"input3": [1, 2, 3], "input4": {"key": "value"}},
                },
                {
                    "component1": {
                        "input1": [{"sender": None, "value": 42}],
                        "input2": [{"sender": None, "value": "test"}],
                    },
                    "component2": {
                        "input3": [{"sender": None, "value": [1, 2, 3]}],
                        "input4": [{"sender": None, "value": {"key": "value"}}],
                    },
                },
            ),
        ],
        ids=["empty_input", "single_component_multiple_inputs", "multiple_components"],
    )
    def test__convert_to_internal_format(self, pipeline_inputs, expected_output):
        """Test conversion of legacy pipeline inputs to internal format."""
        result = PipelineBase._convert_to_internal_format(pipeline_inputs)
        assert result == expected_output

    @pytest.mark.parametrize(
        "socket_type,existing_inputs,expected_count",
        [
            ("regular", None, 1),  # Regular socket should overwrite
            ("regular", [{"sender": "other", "value": 24}], 1),  # Should still overwrite
            ("lazy_variadic", None, 1),  # First input to lazy variadic
            ("lazy_variadic", [{"sender": "other", "value": 24}], 2),  # Should append
        ],
        ids=["regular-new", "regular-existing", "variadic-new", "variadic-existing"],
    )
    def test__write_component_outputs_different_sockets(
        self,
        socket_type,
        existing_inputs,
        expected_count,
        regular_output_socket,
        regular_input_socket,
        lazy_variadic_input_socket,
    ):
        """Test writing to different socket types with various existing input states"""
        receiver_socket = lazy_variadic_input_socket if socket_type == "lazy_variadic" else regular_input_socket
        socket_name = receiver_socket.name
        receivers = [("receiver1", regular_output_socket, receiver_socket)]

        inputs = {}
        if existing_inputs:
            inputs = {"receiver1": {socket_name: existing_inputs}}

        component_outputs = {"output1": 42}

        PipelineBase._write_component_outputs(
            component_name="sender1",
            component_outputs=component_outputs,
            inputs=inputs,
            receivers=receivers,
            include_outputs_from=[],
        )

        assert len(inputs["receiver1"][socket_name]) == expected_count
        assert {"sender": "sender1", "value": 42} in inputs["receiver1"][socket_name]

    @pytest.mark.parametrize(
        "component_outputs,include_outputs,expected_pruned",
        [
            ({"output1": 42, "output2": 24}, [], {"output2": 24}),  # Prune consumed outputs only
            ({"output1": 42, "output2": 24}, ["sender1"], {"output1": 42, "output2": 24}),  # Keep all outputs
            ({}, [], {}),  # No outputs case
        ],
        ids=["prune-consumed", "keep-all", "no-outputs"],
    )
    def test__write_component_outputs_output_pruning(
        self, component_outputs, include_outputs, expected_pruned, regular_output_socket, regular_input_socket
    ):
        """Test output pruning behavior under different scenarios"""
        receivers = [("receiver1", regular_output_socket, regular_input_socket)]

        pruned_outputs = PipelineBase._write_component_outputs(
            component_name="sender1",
            component_outputs=component_outputs,
            inputs={},
            receivers=receivers,
            include_outputs_from=include_outputs,
        )

        assert pruned_outputs == expected_pruned

    @pytest.mark.parametrize(
        "output_value",
        [42, None, _NO_OUTPUT_PRODUCED, "string_value", 3.14],
        ids=["int", "none", "no-output", "string", "float"],
    )
    def test__write_component_outputs_different_output_values(
        self, output_value, regular_output_socket, regular_input_socket
    ):
        """Test handling of different output values"""
        receivers = [("receiver1", regular_output_socket, regular_input_socket)]
        component_outputs = {"output1": output_value}
        inputs = {}
        PipelineBase._write_component_outputs(
            component_name="sender1",
            component_outputs=component_outputs,
            inputs=inputs,
            receivers=receivers,
            include_outputs_from=[],
        )

        assert inputs["receiver1"]["input1"] == [{"sender": "sender1", "value": output_value}]

    def test__write_component_outputs_dont_overwrite_with_no_output(self, regular_output_socket, regular_input_socket):
        """Test that existing inputs are not overwritten with _NO_OUTPUT_PRODUCED"""
        receivers = [("receiver1", regular_output_socket, regular_input_socket)]
        component_outputs = {"output1": _NO_OUTPUT_PRODUCED}
        inputs = {"receiver1": {"input1": [{"sender": "sender1", "value": "keep"}]}}
        PipelineBase._write_component_outputs(
            component_name="sender1",
            component_outputs=component_outputs,
            inputs=inputs,
            receivers=receivers,
            include_outputs_from=[],
        )

        assert inputs["receiver1"]["input1"] == [{"sender": "sender1", "value": "keep"}]

    @pytest.mark.parametrize("receivers_count", [1, 2, 3], ids=["single-receiver", "two-receivers", "three-receivers"])
    def test__write_component_outputs_multiple_receivers(
        self, receivers_count, regular_output_socket, regular_input_socket
    ):
        """Test writing to multiple receivers"""
        receivers = [(f"receiver{i}", regular_output_socket, regular_input_socket) for i in range(receivers_count)]
        component_outputs = {"output1": 42}

        inputs = {}
        PipelineBase._write_component_outputs(
            component_name="sender1",
            component_outputs=component_outputs,
            inputs=inputs,
            receivers=receivers,
            include_outputs_from=[],
        )

        for i in range(receivers_count):
            receiver_name = f"receiver{i}"
            assert receiver_name in inputs
            assert inputs[receiver_name]["input1"] == [{"sender": "sender1", "value": 42}]

    def test__get_next_runnable_component_empty(self):
        """Test with empty queue returns None"""
        queue = FIFOPriorityQueue()
        pipeline = PipelineBase()
        result = pipeline._get_next_runnable_component(queue, component_visits={})
        assert result is None

    def test__get_next_runnable_component_blocked(self):
        """Test component with BLOCKED priority returns None"""
        pipeline = PipelineBase()
        queue = FIFOPriorityQueue()
        queue.push("blocked_component", ComponentPriority.BLOCKED)
        result = pipeline._get_next_runnable_component(queue, component_visits={"blocked_component": 0})
        assert result is None

    @patch("haystack.core.pipeline.base.PipelineBase._get_component_with_graph_metadata_and_visits")
    def test__get_next_runnable_component_max_visits(self, mock_get_component_with_graph_metadata_and_visits):
        """Test component exceeding max visits raises exception"""
        pipeline = PipelineBase(max_runs_per_component=2)
        queue = FIFOPriorityQueue()
        queue.push("ready_component", ComponentPriority.READY)
        mock_get_component_with_graph_metadata_and_visits.return_value = {"instance": "test", "visits": 3}

        with pytest.raises(PipelineMaxComponentRuns) as exc_info:
            pipeline._get_next_runnable_component(queue, component_visits={"ready_component": 3})

        assert "Maximum run count 2 reached for component 'ready_component'" in str(exc_info.value)

    @patch("haystack.core.pipeline.base.PipelineBase._get_component_with_graph_metadata_and_visits")
    def test__get_next_runnable_component_ready(self, mock_get_component_with_graph_metadata_and_visits):
        """Test component that is READY"""
        pipeline = PipelineBase()
        queue = FIFOPriorityQueue()
        queue.push("ready_component", ComponentPriority.READY)
        mock_get_component_with_graph_metadata_and_visits.return_value = {"instance": "test", "visits": 1}

        priority, component_name, component = pipeline._get_next_runnable_component(
            queue, component_visits={"ready_component": 1}
        )

        assert priority == ComponentPriority.READY
        assert component_name == "ready_component"
        assert component == {"instance": "test", "visits": 1}

    @pytest.mark.parametrize(
        "queue_setup,expected_stale",
        [
            # Empty queue case
            (None, True),
            # READY priority case
            ((ComponentPriority.READY, "component1"), False),
            # DEFER priority case
            ((ComponentPriority.DEFER, "component1"), True),
        ],
        ids=["empty-queue", "ready-component", "deferred-component"],
    )
    def test__is_queue_stale(self, queue_setup, expected_stale):
        queue = FIFOPriorityQueue()
        if queue_setup:
            priority, component_name = queue_setup
            queue.push(component_name, priority)

        result = PipelineBase._is_queue_stale(queue)
        assert result == expected_stale

    @patch("haystack.core.pipeline.base.PipelineBase._calculate_priority")
    @patch("haystack.core.pipeline.base.PipelineBase._get_component_with_graph_metadata_and_visits")
    def test_fill_queue(self, mock_get_metadata, mock_calc_priority):
        pipeline = PipelineBase()
        component_names = ["comp1", "comp2"]
        inputs = {"comp1": {"input1": "value1"}, "comp2": {"input2": "value2"}}

        mock_get_metadata.side_effect = lambda name, _: {"component": f"mock_{name}"}
        mock_calc_priority.side_effect = [1, 2]  # Different priorities for testing

        queue = pipeline._fill_queue(component_names, inputs, component_visits={"comp1": 1, "comp2": 1})

        assert mock_get_metadata.call_count == 2
        assert mock_calc_priority.call_count == 2

        # Verify correct calls for first component
        mock_get_metadata.assert_any_call("comp1", 1)
        mock_calc_priority.assert_any_call({"component": "mock_comp1"}, {"input1": "value1"})

        # Verify correct calls for second component
        mock_get_metadata.assert_any_call("comp2", 1)
        mock_calc_priority.assert_any_call({"component": "mock_comp2"}, {"input2": "value2"})

        assert queue.pop() == (1, "comp1")
        assert queue.pop() == (2, "comp2")

    @pytest.mark.parametrize(
        "input_sockets,component_inputs,expected_consumed,expected_remaining",
        [
            # Regular socket test
            (
                {"input1": InputSocket("input1", int)},
                {"input1": [{"sender": "comp1", "value": 42}, {"sender": "comp2", "value": 24}]},
                {"input1": 42},  # Should take first valid input
                {},  # All pipeline inputs should be removed
            ),
            # Regular socket with user input
            (
                {"input1": InputSocket("input1", int)},
                {
                    "input1": [
                        {"sender": "comp1", "value": 42},
                        {"sender": None, "value": 24},  # User input
                    ]
                },
                {"input1": 42},
                {"input1": [{"sender": None, "value": 24}]},  # User input should remain
            ),
            # Greedy variadic socket
            (
                {"greedy": InputSocket("greedy", GreedyVariadic[int])},
                {
                    "greedy": [
                        {"sender": "comp1", "value": 42},
                        {"sender": None, "value": 24},  # User input
                        {"sender": "comp2", "value": 33},
                    ]
                },
                {"greedy": [42]},  # Takes first valid input
                {},  # All inputs removed for greedy sockets
            ),
            # Lazy variadic socket
            (
                {"lazy": InputSocket("lazy", Variadic[int])},
                {
                    "lazy": [
                        {"sender": "comp1", "value": 42},
                        {"sender": "comp2", "value": 24},
                        {"sender": None, "value": 33},  # User input
                    ]
                },
                {"lazy": [42, 24, 33]},  # Takes all valid inputs
                {"lazy": [{"sender": None, "value": 33}]},  # User input remains
            ),
            # Mixed socket types
            (
                {
                    "regular": InputSocket("regular", int),
                    "greedy": InputSocket("greedy", GreedyVariadic[int]),
                    "lazy": InputSocket("lazy", Variadic[int]),
                },
                {
                    "regular": [{"sender": "comp1", "value": 42}, {"sender": None, "value": 24}],
                    "greedy": [{"sender": "comp2", "value": 33}, {"sender": None, "value": 15}],
                    "lazy": [{"sender": "comp3", "value": 55}, {"sender": "comp4", "value": 66}],
                },
                {"regular": 42, "greedy": [33], "lazy": [55, 66]},
                {"regular": [{"sender": None, "value": 24}]},  # Only non-greedy user input remains
            ),
            # Filtering _NO_OUTPUT_PRODUCED
            (
                {"input1": InputSocket("input1", int)},
                {
                    "input1": [
                        {"sender": "comp1", "value": _NO_OUTPUT_PRODUCED},
                        {"sender": "comp2", "value": 42},
                        {"sender": "comp2", "value": _NO_OUTPUT_PRODUCED},
                    ]
                },
                {"input1": 42},  # Should skip _NO_OUTPUT_PRODUCED values
                {},  # All inputs consumed
            ),
        ],
        ids=[
            "regular-socket",
            "regular-with-user-input",
            "greedy-variadic",
            "lazy-variadic",
            "mixed-sockets",
            "no-output-filtering",
        ],
    )
    def test__consume_component_inputs(self, input_sockets, component_inputs, expected_consumed, expected_remaining):
        # Setup
        component = {"input_sockets": input_sockets}
        inputs = {"test_component": component_inputs}

        # Run
        consumed = PipelineBase._consume_component_inputs("test_component", component, inputs)

        # Verify
        assert consumed == expected_consumed
        assert inputs["test_component"] == expected_remaining

    def test__consume_component_inputs_with_df(self, regular_input_socket):
        component = {"input_sockets": {"input1": regular_input_socket}}
        inputs = {"test_component": {"input1": [{"sender": "sender1", "value": DataFrame({"a": [1, 2], "b": [1, 2]})}]}}

        consumed = PipelineBase._consume_component_inputs("test_component", component, inputs)

        assert consumed["input1"].equals(DataFrame({"a": [1, 2], "b": [1, 2]}))

    @patch("haystack.core.pipeline.draw.requests")
    def test_pipeline_draw_called_with_positional_args_triggers_a_warning(self, mock_requests):
        """
        Test that calling the pipeline draw method with positional arguments raises a warning.
        """
        from pathlib import Path
        import warnings

        pipeline = PipelineBase()
        mock_response = mock_requests.get.return_value
        mock_response.status_code = 200
        mock_response.content = b"image_data"
        out_file = Path("original_pipeline.png")
        with warnings.catch_warnings(record=True) as w:
            pipeline.draw(out_file, server_url="http://localhost:3000")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert (
                "Warning: In an upcoming release, this method will require keyword arguments for all parameters"
                in str(w[0].message)
            )

    @patch("haystack.core.pipeline.draw.requests")
    @patch("haystack.core.pipeline.base.is_in_jupyter")
    def test_pipeline_show_called_with_positional_args_triggers_a_warning(self, mock_is_in_jupyter, mock_requests):
        """
        Test that calling the pipeline show method with positional arguments raises a warning.
        """
        import warnings

        pipeline = PipelineBase()
        mock_response = mock_requests.get.return_value
        mock_response.status_code = 200
        mock_response.content = b"image_data"
        mock_is_in_jupyter.return_value = True

        with warnings.catch_warnings(record=True) as w:
            pipeline.show("http://localhost:3000")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert (
                "Warning: In an upcoming release, this method will require keyword arguments for all parameters"
                in str(w[0].message)
            )

    @patch("haystack.core.pipeline.draw.requests")
    def test_pipeline_draw_called_with_keyword_args_triggers_no_warning(self, mock_requests):
        """
        Test that calling the pipeline draw method with keyword arguments does not raise a warning.
        """
        from pathlib import Path
        import warnings

        pipeline = PipelineBase()
        mock_response = mock_requests.get.return_value
        mock_response.status_code = 200
        mock_response.content = b"image_data"
        out_file = Path("original_pipeline.png")

        with warnings.catch_warnings(record=True) as w:
            pipeline.draw(path=out_file, server_url="http://localhost:3000")
            assert len(w) == 0, "No warning should be triggered when using keyword arguments"

    @patch("haystack.core.pipeline.draw.requests")
    @patch("haystack.core.pipeline.base.is_in_jupyter")
    def test_pipeline_show_called_with_keyword_args_triggers_no_warning(self, mock_is_in_jupyter, mock_requests):
        """
        Test that calling the pipeline show method with keyword arguments does not raise a warning.
        """
        import warnings

        pipeline = PipelineBase()
        mock_response = mock_requests.get.return_value
        mock_response.status_code = 200
        mock_response.content = b"image_data"
        mock_is_in_jupyter.return_value = True

        with warnings.catch_warnings(record=True) as w:
            pipeline.show(server_url="http://localhost:3000")
            assert len(w) == 0, "No warning should be triggered when using keyword arguments"


class TestValidateInput:
    def test_validate_input_valid_data(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", input_types={"x": int}, output_types={"y": int})()
        pipe.add_component("comp1", comp1)
        pipe._validate_input(data={"comp1": {"x": 1}})
        # No exception should be raised

    def test_validate_input_missing_mandatory_input(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", input_types={"x": int}, output_types={"y": int})()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError, match="Missing mandatory input 'x' for component 'comp1'"):
            pipe._validate_input(data={"comp1": {}})

    def test_validate_input_missing_mandatory_input_for_component_not_in_data(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", input_types={"x": int}, output_types={"y": int})()
        comp2 = component_class("Comp2", input_types={"a": str}, output_types={"b": str})()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2) # comp2 requires 'a' but is not in data
        with pytest.raises(ValueError, match="Missing mandatory input 'a' for component 'comp2' which was not provided in the input data."):
            pipe._validate_input(data={"comp1": {"x": 1}})


    def test_validate_input_to_already_connected_socket(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", input_types={"x": int}, output_types={"y": int})()
        comp2 = component_class("Comp2", input_types={"a": int}, output_types={"b": int})()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.y", "comp2.a")
        with pytest.raises(ValueError, match="Input 'a' for component 'comp2' is already provided by component 'comp1'. Do not provide it directly."):
            pipe._validate_input(data={"comp2": {"a": 1}})

    def test_validate_input_for_non_existent_component(self):
        pipe = PipelineBase()
        with pytest.raises(ValueError, match="Component 'non_existent' not found in the pipeline. Available components: \\[\\]"):
            pipe._validate_input(data={"non_existent": {"x": 1}})

    def test_validate_input_with_unexpected_input_name(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", input_types={"x": int}, output_types={"y": int})()
        pipe.add_component("comp1", comp1)
        with pytest.raises(ValueError, match="Unexpected input 'z' for component 'comp1'. Available inputs: \\['x'\\]"):
            pipe._validate_input(data={"comp1": {"z": 1}})

    def test_validate_input_variadic_socket_can_receive_multiple_inputs(self):
        pipe = PipelineBase()
        comp1 = component_class("Comp1", output_types={"y": int})()
        comp2 = component_class("Comp2", input_types={"a": Variadic[int]}, output_types={"b": int})()
        pipe.add_component("comp1", comp1)
        pipe.add_component("comp2", comp2)
        pipe.connect("comp1.y", "comp2.a")
        # Should not raise an error, as variadic sockets can accept multiple inputs
        pipe._validate_input(data={"comp2": {"a": 1}})
