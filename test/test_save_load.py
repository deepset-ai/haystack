# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from canals import Pipeline, marshal_pipelines, unmarshal_pipelines
from canals.pipeline.save_load import _rename_connections, _remove_duplicate_instances, _cleanup_marshalled_data
from sample_components import AddFixedValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_marshal_pipelines():
    add_1 = AddFixedValue(add=200)
    add_2 = AddFixedValue()

    pipeline_1 = Pipeline(metadata={"type": "test pipeline", "author": "me"})
    pipeline_1.add_component("first_addition", add_1)
    pipeline_1.add_component("double", Double())
    pipeline_1.add_component("second_addition", add_2)
    pipeline_1.add_component("third_addition", add_1)

    pipeline_1.connect("first_addition.result", "double.value")
    pipeline_1.connect("double.value", "second_addition.value")
    pipeline_1.connect("second_addition.result", "third_addition.value")

    pipeline_2 = Pipeline(metadata={"type": "another test pipeline", "author": "you"})
    pipeline_2.add_component("adder", add_1)
    pipeline_2.add_component("second_addition", add_2)

    pipeline_2.connect("adder.result", "second_addition.value")

    res = marshal_pipelines(pipelines={"pipe1": pipeline_1, "pipe2": pipeline_2})
    assert res == {
        "components": {
            "first_addition": {
                "type": "AddFixedValue",
                "init_parameters": {"add": 200},
            },
            "double": {"type": "Double", "init_parameters": {}},
            "second_addition": {
                "type": "AddFixedValue",
                "init_parameters": {},
            },
            "third_addition": "first_addition",
            "adder": "first_addition",
        },
        "pipelines": {
            "pipe1": {
                "metadata": {"type": "test pipeline", "author": "me"},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "first_addition.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "second_addition.value"},
                    {"sender": "second_addition.result", "receiver": "third_addition.value"},
                ],
            },
            "pipe2": {
                "metadata": {"type": "another test pipeline", "author": "you"},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "adder.result", "receiver": "second_addition.value"},
                ],
            },
        },
    }


def test_marshal_pipelines_with_reused_component_instance_and_identical_names():
    add_two = AddFixedValue(add=2)
    # another_add_two = AddFixedValue(add=2)
    another_add_two = Double()
    double = Double()

    first_pipeline = Pipeline()
    first_pipeline.add_component("add_two", add_two)
    first_pipeline.add_component("double", double)
    first_pipeline.connect("add_two", "double")

    second_pipeline = Pipeline()
    second_pipeline.add_component("add_two", another_add_two)
    second_pipeline.add_component("double", double)
    second_pipeline.connect("add_two", "double")

    res = marshal_pipelines({"first_pipeline": first_pipeline, "second_pipeline": second_pipeline})
    assert res == {
        "components": {
            "add_two_2": {
                "type": "Double",
                "init_parameters": {},
            },
            "add_two_1": {
                "type": "AddFixedValue",
                "init_parameters": {"add": 2},
            },
            "double": {"type": "Double", "init_parameters": {}},
        },
        "pipelines": {
            "first_pipeline": {
                "metadata": {},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "add_two_1.result", "receiver": "double.value"},
                ],
            },
            "second_pipeline": {
                "metadata": {},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "add_two_2.value", "receiver": "double.value"},
                ],
            },
        },
    }


def test_marshal_pipelines_with_reused_component_with_different_names():
    add_two = AddFixedValue(add=2)
    another_add_two = AddFixedValue(add=2)
    double = Double()

    first_pipeline = Pipeline()
    first_pipeline.add_component("add_two", add_two)
    first_pipeline.add_component("double", double)
    first_pipeline.add_component("another_add_two", another_add_two)
    first_pipeline.connect("add_two", "double")
    first_pipeline.connect("double", "another_add_two")

    second_pipeline = Pipeline()
    second_pipeline.add_component("add_two", another_add_two)
    second_pipeline.add_component("double", double)
    second_pipeline.connect("add_two", "double")

    res = marshal_pipelines({"first_pipeline": first_pipeline, "second_pipeline": second_pipeline})
    assert res == {
        "components": {
            "add_two_1": {
                "type": "AddFixedValue",
                "init_parameters": {"add": 2},
            },
            "add_two_2": {
                "type": "AddFixedValue",
                "init_parameters": {"add": 2},
            },
            "double": {"type": "Double", "init_parameters": {}},
            "another_add_two": "add_two_2",
        },
        "pipelines": {
            "first_pipeline": {
                "metadata": {},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "add_two_1.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "another_add_two.value"},
                ],
            },
            "second_pipeline": {
                "metadata": {},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "add_two_2.result", "receiver": "double.value"},
                ],
            },
        },
    }


def test_rename_connections():
    data = {
        "components": {
            "comp1": {"hash": 12},
            "comp2": {"hash": 13},
            "comp3": {"hash": 10},
            "comp4": {"hash": 11},
        },
        "pipelines": {
            "first": {
                "components": {
                    "to_rename": {"hash": 10},
                    "comp1": {"hash": 12},
                },
                "connections": [
                    {
                        "sender": "to_rename.output",
                        "receiver": "comp1.input",
                    }
                ],
            },
            "second": {
                "components": {
                    "to_rename": {"hash": 11},
                    "comp2": {"hash": 13},
                },
                "connections": [
                    {
                        "sender": "to_rename.output",
                        "receiver": "comp2.input",
                    }
                ],
            },
        },
    }
    renames = {
        "comp3": "to_rename",
        "comp4": "to_rename",
    }
    _rename_connections(data, renames)
    assert data == {
        "components": {
            "comp1": {"hash": 12},
            "comp2": {"hash": 13},
            "comp3": {"hash": 10},
            "comp4": {"hash": 11},
        },
        "pipelines": {
            "first": {
                "components": {
                    "to_rename": {"hash": 10},
                    "comp1": {"hash": 12},
                },
                "connections": [
                    {
                        "sender": "comp3.output",
                        "receiver": "comp1.input",
                    }
                ],
            },
            "second": {
                "components": {
                    "to_rename": {"hash": 11},
                    "comp2": {"hash": 13},
                },
                "connections": [
                    {
                        "sender": "comp4.output",
                        "receiver": "comp2.input",
                    }
                ],
            },
        },
    }


def test_remove_duplicate_instances():
    components = {
        "comp1": {"hash": 123},
        "comp2": {"hash": 123},
        "comp3": {"hash": 456},
    }
    _remove_duplicate_instances(components)
    assert components == {
        "comp1": {"hash": 123},
        "comp2": "comp1",
        "comp3": {"hash": 456},
    }


def test_cleanup_marshalled_data():
    data = {
        "components": {
            "comp1": {"hash": 123},
            "comp2": {"hash": 123},
            "comp3": {"hash": 456},
        },
        "pipelines": {
            "first": {
                "components": {
                    "comp1": {},
                }
            },
            "second": {
                "components": {
                    "comp2": {},
                    "comp3": {},
                }
            },
        },
    }
    _cleanup_marshalled_data(data)
    assert data == {
        "components": {
            "comp1": {},
            "comp2": {},
            "comp3": {},
        },
        "pipelines": {
            "first": {},
            "second": {},
        },
    }


@pytest.mark.skip
def test_unmarshal_pipelines():
    data = {
        "components": {
            "first_addition": {
                "type": "AddFixedValue",
                "init_parameters": {"add": 200},
            },
            "double": {"type": "Double", "init_parameters": {}},
            "second_addition": {
                "type": "AddFixedValue",
                "init_parameters": {},
            },
            "third_addition": "first_addition",
        },
        "pipelines": {
            "pipe1": {
                "metadata": {"type": "test pipeline", "author": "me"},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "first_addition.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "second_addition.value"},
                    {"sender": "second_addition.result", "receiver": "third_addition.value"},
                ],
            },
            "pipe2": {
                "metadata": {"type": "another test pipeline", "author": "you"},
                "max_loops_allowed": 100,
                "connections": [
                    {"sender": "first_addition.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "second_addition.value"},
                ],
            },
        },
    }
    pipelines = unmarshal_pipelines(data)

    pipe1 = pipelines["pipe1"]
    assert pipe1.metadata == {"type": "test pipeline", "author": "me"}

    first_addition = pipe1.get_component("first_addition")
    assert type(first_addition) == AddFixedValue
    assert pipe1.graph.nodes["first_addition"]["instance"].add == 300

    second_addition = pipe1.get_component("second_addition")
    assert type(second_addition) == AddFixedValue
    assert pipe1.graph.nodes["second_addition"]["instance"].add == 1
    assert second_addition != first_addition

    third_addition = pipe1.get_component("third_addition")
    assert type(third_addition) == AddFixedValue
    assert pipe1.graph.nodes["third_addition"]["instance"].add == 300
    assert third_addition == first_addition

    double = pipe1.get_component("double")
    assert type(double) == Double

    assert list(pipe1.graph.edges) == [
        ("first_addition", "double", "result/value"),
        ("double", "second_addition", "value/value"),
        ("second_addition", "third_addition", "result/value"),
    ]

    pipe2 = pipelines["pipe2"]
    assert pipe2.metadata == {"type": "another test pipeline", "author": "you"}

    first_addition_2 = pipe2.get_component("first_addition")
    assert type(first_addition_2) == AddFixedValue
    assert pipe2.graph.nodes["first_addition"]["instance"].add == 300
    assert first_addition_2 == first_addition

    second_addition_2 = pipe2.get_component("second_addition")
    assert type(second_addition_2) == AddFixedValue
    assert pipe2.graph.nodes["second_addition"]["instance"].add == 1
    assert second_addition_2 != first_addition_2
    assert second_addition_2 == second_addition

    with pytest.raises(ValueError):
        pipe2.get_component("third_addition")

    double_2 = pipe2.get_component("double")
    assert type(double_2) == Double
    assert double_2 != double

    assert list(pipe2.graph.edges) == [
        ("first_addition", "double", "result/value"),
        ("double", "second_addition", "value/value"),
    ]
