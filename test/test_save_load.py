# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals import Pipeline, marshal_pipelines, unmarshal_pipelines
from canals.pipeline.save_load import (
    _rename_connections,
    _remove_duplicate_instances,
    _remove_pipeline_component_data,
    _remove_component_hashes,
)
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
    another_double = Double()
    double = Double()

    first_pipeline = Pipeline()
    first_pipeline.add_component("add_two", add_two)
    first_pipeline.add_component("double", double)
    first_pipeline.connect("add_two", "double")

    second_pipeline = Pipeline()
    second_pipeline.add_component("add_two", another_double)
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


def test_remove_pipeline_component_data():
    data = {
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
    _remove_pipeline_component_data(data)
    assert data == {
        "pipelines": {
            "first": {},
            "second": {},
        },
    }


def test_remove_component_hashes():
    data = {
        "components": {
            "comp1": {"hash": 123},
            "comp2": {"hash": 123},
            "comp3": {"hash": 456},
        },
    }
    _remove_component_hashes(data)
    assert data == {
        "components": {
            "comp1": {},
            "comp2": {},
            "comp3": {},
        },
    }


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
                "max_loops_allowed": 42,
                "connections": [
                    {"sender": "first_addition.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "second_addition.value"},
                    {"sender": "second_addition.result", "receiver": "third_addition.value"},
                ],
            },
            "pipe2": {
                "metadata": {"author": "you"},
                "connections": [
                    {"sender": "first_addition.result", "receiver": "double.value"},
                    {"sender": "double.value", "receiver": "second_addition.value"},
                ],
            },
        },
    }
    pipelines = unmarshal_pipelines(data)
    assert len(pipelines) == 2

    pipe1 = pipelines["pipe1"]
    assert pipe1.metadata == {"type": "test pipeline", "author": "me"}
    assert pipe1.max_loops_allowed == 42

    components1 = dict(pipe1.graph.nodes(data="instance"))
    assert len(components1) == 4

    assert isinstance(components1["first_addition"], AddFixedValue)
    assert components1["first_addition"].add == 200

    assert isinstance(components1["double"], Double)

    assert isinstance(components1["second_addition"], AddFixedValue)
    assert components1["second_addition"].add == 1

    assert isinstance(components1["third_addition"], AddFixedValue)
    assert components1["third_addition"].add == 200
    assert components1["third_addition"] is components1["first_addition"]

    connections1 = list(pipe1.graph.edges)
    assert len(connections1) == 3
    assert connections1[0] == ("first_addition", "double", "result/value")
    assert connections1[1] == ("double", "second_addition", "value/value")
    assert connections1[2] == ("second_addition", "third_addition", "result/value")

    pipe2 = pipelines["pipe2"]
    assert pipe2.metadata == {"author": "you"}
    assert pipe2.max_loops_allowed == 100

    components2 = dict(pipe2.graph.nodes(data="instance"))
    assert len(components2) == 3

    assert isinstance(components2["first_addition"], AddFixedValue)
    assert components2["first_addition"].add == 200

    assert isinstance(components1["double"], Double)

    assert isinstance(components1["second_addition"], AddFixedValue)
    assert components1["second_addition"].add == 1

    connections2 = list(pipe2.graph.edges)
    assert len(connections2) == 2
    assert connections2[0] == ("first_addition", "double", "result/value")
    assert connections2[1] == ("double", "second_addition", "value/value")


def test_unmarshal_pipelines_just_marshalled():
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

    data = marshal_pipelines({"first_pipeline": first_pipeline, "second_pipeline": second_pipeline})
    res = unmarshal_pipelines(data)

    assert len(res) == 2

    assert res["first_pipeline"] != first_pipeline
    first_components = dict(res["first_pipeline"].graph.nodes(data="instance"))
    assert len(first_components) == 3

    assert isinstance(first_components["add_two_1"], AddFixedValue)
    assert first_components["add_two_1"].add == 2

    assert isinstance(first_components["double"], Double)

    assert isinstance(first_components["another_add_two"], AddFixedValue)
    assert first_components["add_two_1"].add == 2

    first_connections = list(res["first_pipeline"].graph.edges)
    assert len(first_connections) == 2
    assert first_connections[0] == ("add_two_1", "double", "result/value")
    assert first_connections[1] == ("double", "another_add_two", "value/value")

    assert res["second_pipeline"] != second_pipeline
    second_components = dict(res["second_pipeline"].graph.nodes(data="instance"))
    assert len(second_components) == 2

    assert isinstance(second_components["add_two_2"], AddFixedValue)
    assert second_components["add_two_2"].add == 2

    assert isinstance(second_components["double"], Double)

    assert second_components["add_two_2"] is first_components["another_add_two"]

    second_connections = list(res["second_pipeline"].graph.edges)
    assert len(second_connections) == 1
    assert second_connections[0] == ("add_two_2", "double", "result/value")
