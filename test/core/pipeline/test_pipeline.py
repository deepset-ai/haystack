# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.components.builders import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.others import Multiplexer
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers import ConditionalRouter
from haystack.core.component import component
from haystack.core.component.types import InputSocket, OutputSocket
from haystack.core.errors import PipelineDrawingError, PipelineError, PipelineMaxLoops, PipelineRuntimeError
from haystack.core.pipeline import Pipeline, PredefinedPipeline
from haystack.core.serialization import DeserializationCallbacks
from haystack.document_stores.in_memory import InMemoryDocumentStore
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


def test_run_with_greedy_variadic_after_component_with_default_input_simple(spying_tracer):
    """
    This test verifies that `Pipeline.run()` executes the components in the correct order when
    there's a greedy Component with variadic input right before a Component with at least one default input.

    We use the `spying_tracer` fixture to simplify the code to verify the order of execution.
    This creates some coupling between this test and how we trace the Pipeline execution.
    A worthy tradeoff in my opinion, we will notice right away if we change either the run logic or
    the tracing logic.
    """
    document_store = InMemoryDocumentStore()
    document_store.write_documents([Document(content="This is a simple document")])

    pipeline = Pipeline()
    template = "Given this documents: {{ documents|join(', ', attribute='content') }} Answer this question: {{ query }}"
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("multiplexer", Multiplexer(List[Document]))

    pipeline.connect("retriever", "multiplexer")
    pipeline.connect("multiplexer", "prompt_builder.documents")
    res = pipeline.run({"query": "This is my question"})

    assert res == {
        "prompt_builder": {
            "prompt": "Given this documents: This is a simple document Answer this question: This is my question"
        }
    }

    assert len(spying_tracer.spans) == 4
    assert spying_tracer.spans[0].operation_name == "haystack.pipeline.run"
    assert spying_tracer.spans[1].operation_name == "haystack.component.run"
    assert spying_tracer.spans[1].tags["haystack.component.name"] == "retriever"
    assert spying_tracer.spans[2].operation_name == "haystack.component.run"
    assert spying_tracer.spans[2].tags["haystack.component.name"] == "multiplexer"
    assert spying_tracer.spans[3].operation_name == "haystack.component.run"
    assert spying_tracer.spans[3].tags["haystack.component.name"] == "prompt_builder"


def test_pipeline_resolution_simple_input():
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, word: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {word}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.word")
    result = pipeline.run(data={"hello": {"word": "world"}})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}

    result = pipeline.run(data={"word": "world"})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}


def test_pipeline_resolution_wrong_input_name(caplog):
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {who}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.who")

    # test case with nested component inputs
    with pytest.raises(ValueError):
        pipeline.run(data={"hello": {"non_existing_input": "world"}})

    # test case with flat component inputs
    with pytest.raises(ValueError):
        pipeline.run(data={"non_existing_input": "world"})

    # important to check that the warning is logged for UX purposes, leave it here
    assert "were not matched to any component" in caplog.text


def test_pipeline_resolution_with_mixed_correct_and_incorrect_input_names(caplog):
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str):
            """
            Takes a string in input and returns "Hello, <string>!"
            in output.
            """
            return {"output": f"Hello, {who}!"}

    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())

    pipeline.connect("hello.output", "hello2.who")

    # test case with nested component inputs
    # this will raise ValueError because hello component does not have an input named "non_existing_input"
    # even though it has an input named "who"
    with pytest.raises(ValueError):
        pipeline.run(data={"hello": {"non_existing_input": "world", "who": "world"}})

    # test case with flat component inputs
    # this will not raise ValueError because the input "who" will be resolved to the correct component
    # and we'll log a warning for the input "non_existing_input" which was not resolved
    result = pipeline.run(data={"non_existing_input": "world", "who": "world"})
    assert result == {"hello2": {"output": "Hello, Hello, world!!"}}

    # important to check that the warning is logged for UX purposes, leave it here
    assert "were not matched to any component" in caplog.text


def test_pipeline_resolution_duplicate_input_names_across_components():
    @component
    class Hello:
        @component.output_types(output=str)
        def run(self, who: str, what: str):
            return {"output": f"Hello {who} {what}!"}

    pipe = Pipeline()
    pipe.add_component("hello", Hello())
    pipe.add_component("hello2", Hello())

    pipe.connect("hello.output", "hello2.who")

    result = pipe.run(data={"what": "Haystack", "who": "world"})
    assert result == {"hello2": {"output": "Hello Hello world Haystack! Haystack!"}}

    resolved, _ = pipe._prepare_component_input_data(data={"what": "Haystack", "who": "world"})

    # why does hello2 have only one input? Because who of hello2 is inserted from hello.output
    assert resolved == {"hello": {"what": "Haystack", "who": "world"}, "hello2": {"what": "Haystack"}}


def test_pipeline_dumps(test_files_path):
    pipeline = Pipeline()
    pipeline.add_component("Comp1", FakeComponent("Foo"))
    pipeline.add_component("Comp2", FakeComponent())
    pipeline.connect("Comp1.value", "Comp2.input_")
    pipeline.max_loops_allowed = 99
    result = pipeline.dumps()
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        assert f.read() == result


def test_pipeline_loads(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.loads(f.read())
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), FakeComponent)
        assert isinstance(pipeline.get_component("Comp2"), FakeComponent)


def test_pipeline_dump(test_files_path, tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("Comp1", FakeComponent("Foo"))
    pipeline.add_component("Comp2", FakeComponent())
    pipeline.connect("Comp1.value", "Comp2.input_")
    pipeline.max_loops_allowed = 99
    with open(tmp_path / "out.yaml", "w") as f:
        pipeline.dump(f)
    # re-open and ensure it's the same data as the test file
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as test_f, open(tmp_path / "out.yaml", "r") as f:
        assert f.read() == test_f.read()


def test_pipeline_load(test_files_path):
    with open(f"{test_files_path}/yaml/test_pipeline.yaml", "r") as f:
        pipeline = Pipeline.load(f)
        assert pipeline.max_loops_allowed == 99
        assert isinstance(pipeline.get_component("Comp1"), FakeComponent)
        assert isinstance(pipeline.get_component("Comp2"), FakeComponent)


@patch("haystack.core.pipeline.base._to_mermaid_image")
@patch("haystack.core.pipeline.base.is_in_jupyter")
@patch("IPython.display.Image")
@patch("IPython.display.display")
def test_show_in_notebook(mock_ipython_display, mock_ipython_image, mock_is_in_jupyter, mock_to_mermaid_image):
    pipe = Pipeline()

    mock_to_mermaid_image.return_value = b"some_image_data"
    mock_is_in_jupyter.return_value = True

    pipe.show()
    mock_ipython_image.assert_called_once_with(b"some_image_data")
    mock_ipython_display.assert_called_once()


@patch("haystack.core.pipeline.base.is_in_jupyter")
def test_show_not_in_notebook(mock_is_in_jupyter):
    pipe = Pipeline()

    mock_is_in_jupyter.return_value = False

    with pytest.raises(PipelineDrawingError):
        pipe.show()


@patch("haystack.core.pipeline.base._to_mermaid_image")
def test_draw(mock_to_mermaid_image, tmp_path):
    pipe = Pipeline()
    mock_to_mermaid_image.return_value = b"some_image_data"

    image_path = tmp_path / "test.png"
    pipe.draw(path=image_path)
    assert image_path.read_bytes() == mock_to_mermaid_image.return_value


def test_add_component_to_different_pipelines():
    first_pipe = Pipeline()
    second_pipe = Pipeline()
    some_component = component_class("Some")()

    assert some_component.__haystack_added_to_pipeline__ is None
    first_pipe.add_component("some", some_component)
    assert some_component.__haystack_added_to_pipeline__ is first_pipe

    with pytest.raises(PipelineError):
        second_pipe.add_component("some", some_component)


def test_get_component_name():
    pipe = Pipeline()
    some_component = component_class("Some")()
    pipe.add_component("some", some_component)

    assert pipe.get_component_name(some_component) == "some"


def test_get_component_name_not_added_to_pipeline():
    pipe = Pipeline()
    some_component = component_class("Some")()

    assert pipe.get_component_name(some_component) == ""


def test_repr():
    pipe = Pipeline(metadata={"test": "test"}, max_loops_allowed=42)
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


def test_run_raises_if_max_visits_reached():
    def custom_init(self):
        component.set_input_type(self, "x", int)
        component.set_input_type(self, "y", int, 1)
        component.set_output_types(self, a=int, b=int)

    FakeComponent = component_class("FakeComponent", output={"a": 1, "b": 1}, extra_fields={"__init__": custom_init})
    pipe = Pipeline(max_loops_allowed=1)
    pipe.add_component("first", FakeComponent())
    pipe.add_component("second", FakeComponent())
    pipe.connect("first.a", "second.x")
    pipe.connect("second.b", "first.y")
    with pytest.raises(PipelineMaxLoops):
        pipe.run({"first": {"x": 1}})


def test_run_with_component_that_does_not_return_dict():
    BrokenComponent = component_class(
        "BrokenComponent", input_types={"a": int}, output_types={"b": int}, output=1  # type:ignore
    )

    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("comp", BrokenComponent())
    with pytest.raises(PipelineRuntimeError):
        pipe.run({"comp": {"a": 1}})


def test_to_dict():
    add_two = AddFixedValue(add=2)
    add_default = AddFixedValue()
    double = Double()
    pipe = Pipeline(metadata={"test": "test"}, max_loops_allowed=42)
    pipe.add_component("add_two", add_two)
    pipe.add_component("add_default", add_default)
    pipe.add_component("double", double)
    pipe.connect("add_two", "double")
    pipe.connect("double", "add_default")

    res = pipe.to_dict()
    expected = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 42,
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


def test_from_dict():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 101,
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
    assert pipe.max_loops_allowed == 101

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


def test_from_dict_with_callbacks():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 101,
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
            "greet": {"type": "haystack.testing.sample_components.greet.Greet", "init_parameters": {"message": "test"}},
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

    pipe = Pipeline.from_dict(data, callbacks=DeserializationCallbacks(component_pre_init=component_pre_init_callback))
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


def test_from_dict_with_empty_dict():
    assert Pipeline() == Pipeline.from_dict({})


def test_from_dict_with_components_instances():
    add_two = AddFixedValue(add=2)
    add_default = AddFixedValue()
    components = {"add_two": add_two, "add_default": add_default}
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
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
    assert pipe.max_loops_allowed == 100

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


def test_from_dict_without_component_type():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {"add_two": {"init_parameters": {"add": 2}}},
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing 'type' in component 'add_two'")


def test_from_dict_without_registered_component_type(request):
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {"add_two": {"type": "foo.bar.baz", "init_parameters": {"add": 2}}},
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match(r"Component .+ not imported.")


def test_from_dict_without_connection_sender():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [{"receiver": "some.receiver"}],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing sender in connection: {'receiver': 'some.receiver'}")


def test_from_dict_without_connection_receiver():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [{"sender": "some.sender"}],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing receiver in connection: {'sender': 'some.sender'}")


def test_falsy_connection():
    A = component_class("A", input_types={"x": int}, output={"y": 0})
    B = component_class("A", input_types={"x": int}, output={"y": 0})
    p = Pipeline()
    p.add_component("a", A())
    p.add_component("b", B())
    p.connect("a.y", "b.x")
    assert p.run({"a": {"x": 10}})["b"]["y"] == 0


def test_describe_input_only_no_inputs_components():
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


def test_describe_input_some_components_with_no_inputs():
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


def test_describe_input_all_components_have_inputs():
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


def test_describe_output_multiple_possible():
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


def test_describe_output_single():
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


def test_describe_no_outputs():
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


def test_from_template(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    with patch("haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore"):
        pipe = Pipeline.from_template(PredefinedPipeline.INDEXING)
        assert pipe.get_component("cleaner")


def test_walk_pipeline_with_no_cycles():
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


def test_walk_pipeline_with_cycles():
    """
    This pipeline consists of one component, which would run three times in a loop.
    pipeline.walk() should return this component exactly once. The order is not guaranteed.
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
    pipeline.add_component("hello", hello)
    pipeline.connect("hello.intermediate", "hello.intermediate")
    assert [("hello", hello)] == list(pipeline.walk())


def test_correct_execution_order_of_components_with_only_defaults(spying_tracer):
    """
    We enqueue the Components in internal `to_run` data structure at the start of `Pipeline.run()` using the order
    they are added in the Pipeline with `Pipeline.add_component()`.
    If a Component A with defaults is added before a Component B that has no defaults, but in the Pipeline
    logic A must be executed after B it could run instead before.

    This test verifies that the order of execution is correct.
    """
    docs = [Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    template = (
        "Given the following information, answer the question.\n"
        "Context:\n"
        "{% for document in documents %}"
        "    {{ document.content }}\n"
        "{% endfor %}"
        "Question: {{ query }}"
    )

    pipe = Pipeline()

    # The order of this addition is important for the test
    # Do not edit them.
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
    pipe.connect("retriever", "prompt_builder.documents")

    query = "What is the capital of France?"
    res = pipe.run({"prompt_builder": {"query": query}, "retriever": {"query": query}})

    assert len(spying_tracer.spans) == 3
    assert spying_tracer.spans[0].operation_name == "haystack.pipeline.run"
    assert spying_tracer.spans[1].operation_name == "haystack.component.run"
    assert spying_tracer.spans[1].tags["haystack.component.name"] == "retriever"
    assert spying_tracer.spans[2].operation_name == "haystack.component.run"
    assert spying_tracer.spans[2].tags["haystack.component.name"] == "prompt_builder"

    print(res["prompt_builder"]["prompt"])
    assert res == {
        "prompt_builder": {
            "prompt": "Given the following information, answer the question.\n"
            "Context:\n"
            "    Paris is the capital of France\n"
            "    Rome is the capital of Italy\n"
            "Question: What is the capital of France?"
        }
    }


def test_pipeline_is_not_stuck_with_components_with_only_defaults():
    FakeGenerator = component_class(
        "FakeGenerator", input_types={"prompt": str}, output_types={"replies": List[str]}, output={"replies": ["Paris"]}
    )
    docs = [Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    template = (
        "Given the following information, answer the question.\n"
        "Context:\n"
        "{% for document in documents %}"
        "    {{ document.content }}\n"
        "{% endfor %}"
        "Question: {{ query }}"
    )

    pipe = Pipeline()

    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=doc_store))
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("generator", FakeGenerator())
    pipe.add_component("answer_builder", AnswerBuilder())

    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.prompt")
    pipe.connect("generator.replies", "answer_builder.replies")
    pipe.connect("retriever.documents", "answer_builder.documents")

    query = "What is the capital of France?"
    res = pipe.run({"query": query})
    assert len(res) == 1
    answers = res["answer_builder"]["answers"]
    assert len(answers) == 1
    assert answers[0].data == "Paris"


def test_pipeline_is_not_stuck_with_components_with_only_defaults_as_first_components():
    """
    This tests verifies that a Pipeline doesn't get stuck running in a loop if
    it has all the following characterics:
    - The first Component has all defaults for its inputs
    - The first Component receives one input from the user
    - The first Component receives one input from a loop in the Pipeline
    - The second Component has at least one default input
    """

    def fake_generator_run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        # Simple hack to simulate a model returning a different reply after the
        # the first time it's called
        if getattr(fake_generator_run, "called", False):
            return {"replies": ["Rome"]}
        fake_generator_run.called = True
        return {"replies": ["Paris"]}

    FakeGenerator = component_class(
        "FakeGenerator",
        input_types={"prompt": str, "generation_kwargs": Optional[Dict[str, Any]]},
        output_types={"replies": List[str]},
        extra_fields={"run": fake_generator_run},
    )
    template = (
        "Answer the following question.\n"
        "{% if previous_replies %}\n"
        "Previously you replied incorrectly this:\n"
        "{% for reply in previous_replies %}\n"
        " - {{ reply }}\n"
        "{% endfor %}\n"
        "{% endif %}\n"
        "Question: {{ query }}"
    )
    router = ConditionalRouter(
        routes=[
            {
                "condition": "{{ replies == ['Rome'] }}",
                "output": "{{ replies }}",
                "output_name": "correct_replies",
                "output_type": List[int],
            },
            {
                "condition": "{{ replies == ['Paris'] }}",
                "output": "{{ replies }}",
                "output_name": "incorrect_replies",
                "output_type": List[int],
            },
        ]
    )

    pipe = Pipeline()

    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    pipe.add_component("generator", FakeGenerator())
    pipe.add_component("router", router)

    pipe.connect("prompt_builder.prompt", "generator.prompt")
    pipe.connect("generator.replies", "router.replies")
    pipe.connect("router.incorrect_replies", "prompt_builder.previous_replies")

    res = pipe.run({"prompt_builder": {"query": "What is the capital of Italy?"}})

    assert res == {"router": {"correct_replies": ["Rome"]}}
