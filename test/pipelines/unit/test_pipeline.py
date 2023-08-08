# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import logging

import pytest

from canals import Pipeline
from canals.pipeline.sockets import InputSocket, OutputSocket
from canals.errors import PipelineMaxLoops, PipelineError
from sample_components import AddFixedValue, Threshold, MergeLoop, Double

logging.basicConfig(level=logging.DEBUG)


def test_max_loops():
    pipe = Pipeline(max_loops_allowed=10)
    pipe.add_component("add", AddFixedValue())
    pipe.add_component("threshold", Threshold(threshold=100))
    pipe.add_component("merge", MergeLoop(expected_type=int, inputs=["value_1", "value_2"]))
    pipe.connect("threshold.below", "add.value")
    pipe.connect("add.result", "merge.value_1")
    pipe.connect("merge.value", "threshold.value")
    with pytest.raises(PipelineMaxLoops):
        pipe.run({"merge": {"value_2": 1}})


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
                "type": "AddFixedValue",
                "hash": id(add_two),
                "init_parameters": {"add": 2},
            },
            "add_default": {
                "type": "AddFixedValue",
                "hash": id(add_default),
                "init_parameters": {},
            },
            "double": {
                "type": "Double",
                "hash": id(double),
                "init_parameters": {},
            },
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
                "type": "AddFixedValue",
                "hash": "123",
                "init_parameters": {"add": 2},
            },
            "add_default": {
                "type": "AddFixedValue",
                "hash": "456",
                "init_parameters": {},
            },
            "double": {
                "type": "Double",
                "hash": "789",
                "init_parameters": {},
            },
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
        "value": InputSocket(name="value", type=int, is_optional=False, sender=None),
        "add": InputSocket(name="add", type=Optional[int], is_optional=True, sender=None),
    }
    assert add_two["output_sockets"] == {
        "result": OutputSocket(name="result", type=int),
    }
    assert add_two["visits"] == 0

    ## add_default
    add_default = pipe.graph.nodes["add_default"]
    assert add_default["instance"].add == 1
    assert add_default["input_sockets"] == {
        "value": InputSocket(name="value", type=int, is_optional=False, sender="double"),
        "add": InputSocket(name="add", type=Optional[int], is_optional=True, sender=None),
    }
    assert add_default["output_sockets"] == {
        "result": OutputSocket(name="result", type=int),
    }
    assert add_default["visits"] == 0

    ## double
    double = pipe.graph.nodes["double"]
    assert double["instance"]
    assert double["input_sockets"] == {
        "value": InputSocket(name="value", type=int, is_optional=False, sender="add_two"),
    }
    assert double["output_sockets"] == {
        "value": OutputSocket(name="value", type=int),
    }
    assert double["visits"] == 0

    # Connections
    connections = list(pipe.graph.edges(data=True))
    assert len(connections) == 2
    assert connections[0] == (
        "add_two",
        "double",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="result", type=int),
            "to_socket": InputSocket(name="value", type=int, is_optional=False, sender="add_two"),
        },
    )
    assert connections[1] == (
        "double",
        "add_default",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="value", type=int),
            "to_socket": InputSocket(name="value", type=int, is_optional=False, sender="double"),
        },
    )


def test_from_dict_with_empty_dict():
    assert Pipeline() == Pipeline.from_dict({})


def test_from_dict_with_components_instances():
    add_two = AddFixedValue(add=2)
    add_default = AddFixedValue()
    components = {
        "add_two": add_two,
        "add_default": add_default,
    }
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {
            "add_two": {},
            "add_default": {},
            "double": {
                "type": "Double",
                "hash": "789",
                "init_parameters": {},
            },
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
        "value": InputSocket(name="value", type=int, is_optional=False, sender=None),
        "add": InputSocket(name="add", type=Optional[int], is_optional=True, sender=None),
    }
    assert add_two_data["output_sockets"] == {
        "result": OutputSocket(name="result", type=int),
    }
    assert add_two_data["visits"] == 0

    ## add_default
    add_default_data = pipe.graph.nodes["add_default"]
    assert add_default_data["instance"] is add_default
    assert add_default_data["instance"].add == 1
    assert add_default_data["input_sockets"] == {
        "value": InputSocket(name="value", type=int, is_optional=False, sender="double"),
        "add": InputSocket(name="add", type=Optional[int], is_optional=True, sender=None),
    }
    assert add_default_data["output_sockets"] == {
        "result": OutputSocket(name="result", type=int),
    }
    assert add_default_data["visits"] == 0

    ## double
    double = pipe.graph.nodes["double"]
    assert double["instance"]
    assert double["input_sockets"] == {
        "value": InputSocket(name="value", type=int, is_optional=False, sender="add_two"),
    }
    assert double["output_sockets"] == {
        "value": OutputSocket(name="value", type=int),
    }
    assert double["visits"] == 0

    # Connections
    connections = list(pipe.graph.edges(data=True))
    assert len(connections) == 2
    assert connections[0] == (
        "add_two",
        "double",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="result", type=int),
            "to_socket": InputSocket(name="value", type=int, is_optional=False, sender="add_two"),
        },
    )
    assert connections[1] == (
        "double",
        "add_default",
        {
            "conn_type": "int",
            "from_socket": OutputSocket(name="value", type=int),
            "to_socket": InputSocket(name="value", type=int, is_optional=False, sender="double"),
        },
    )


def test_from_dict_without_component_type():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {
            "add_two": {
                "hash": "123",
                "init_parameters": {"add": 2},
            },
        },
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing 'type' in component 'add_two'")


def test_from_dict_without_registered_component_type(request):
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {
            "add_two": {
                # We use the test function name as component type to make sure it's not registered.
                "type": request.node.name,
                "hash": "123",
                "init_parameters": {"add": 2},
            },
        },
        "connections": [],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match(f"Component '{request.node.name}' not imported.")


def test_from_dict_without_connection_sender():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [
            {"receiver": "some.receiver"},
        ],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing sender in connection: {'receiver': 'some.receiver'}")


def test_from_dict_without_connection_receiver():
    data = {
        "metadata": {"test": "test"},
        "max_loops_allowed": 100,
        "components": {},
        "connections": [
            {"sender": "some.sender"},
        ],
    }
    with pytest.raises(PipelineError) as err:
        Pipeline.from_dict(data)

    err.match("Missing receiver in connection: {'sender': 'some.sender'}")
