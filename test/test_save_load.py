from pathlib import Path

import pytest

from canals.pipeline import Pipeline, marshal_pipelines, unmarshal_pipelines
from test.nodes import AddValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_marshal():
    add_1 = AddValue(add=1)
    add_2 = AddValue(add=1)

    pipeline_1 = Pipeline(metadata={"type": "test pipeline", "author": "me"})
    pipeline_1.add_node("first_addition", add_2, parameters={"add": 6})
    pipeline_1.add_node("double", Double(input="value"))
    pipeline_1.add_node("second_addition", add_1)
    pipeline_1.add_node("third_addition", add_2)

    pipeline_1.connect("first_addition", "double")
    pipeline_1.connect("double", "second_addition")
    pipeline_1.connect("second_addition", "third_addition")

    pipeline_2 = Pipeline(metadata={"type": "another test pipeline", "author": "you"})
    pipeline_2.add_node("first_addition", add_2, parameters={"add": 4})
    pipeline_2.add_node("double", Double(input="value"))
    pipeline_2.add_node("second_addition", add_1)

    pipeline_2.connect("first_addition", "double")
    pipeline_2.connect("double", "second_addition")

    assert marshal_pipelines(pipelines={"pipe1": pipeline_1, "pipe2": pipeline_2}) == {
        "pipelines": {
            "pipe1": {
                "metadata": {"type": "test pipeline", "author": "me"},
                "max_loops_allowed": 100,
                "nodes": {
                    "first_addition": {
                        "type": "AddValue",
                        "init_parameters": {"add": 1, "input": "value", "output": "value"},
                        "run_parameters": {"add": 6},
                    },
                    "double": {"type": "Double", "init_parameters": {"input": "value", "output": "value"}},
                    "second_addition": {
                        "type": "AddValue",
                        "init_parameters": {"add": 1, "input": "value", "output": "value"},
                    },
                    "third_addition": {"refer_to": "pipe1.first_addition"},
                },
                "edges": [
                    ("first_addition", "double"),
                    ("double", "second_addition"),
                    ("second_addition", "third_addition"),
                ],
            },
            "pipe2": {
                "metadata": {"type": "another test pipeline", "author": "you"},
                "max_loops_allowed": 100,
                "nodes": {
                    "first_addition": {"refer_to": "pipe1.first_addition", "run_parameters": {"add": 4}},
                    "double": {"type": "Double", "init_parameters": {"input": "value", "output": "value"}},
                    "second_addition": {"refer_to": "pipe1.second_addition"},
                },
                "edges": [
                    ("first_addition", "double"),
                    ("double", "second_addition"),
                ],
            },
        },
        "dependencies": ["test", "canals"],
    }


def test_unmarshal():
    pipelines = unmarshal_pipelines(
        {
            "pipelines": {
                "pipe1": {
                    "metadata": {"type": "test pipeline", "author": "me"},
                    "max_loops_allowed": 100,
                    "nodes": {
                        "first_addition": {
                            "type": "AddValue",
                            "init_parameters": {"add": 1, "input": "value", "output": "value"},
                            "run_parameters": {"add": 6},
                        },
                        "double": {"type": "Double", "init_parameters": {"input": "value", "output": "value"}},
                        "second_addition": {
                            "type": "AddValue",
                            "init_parameters": {"add": 1, "input": "value", "output": "value"},
                        },
                        "third_addition": {"refer_to": "pipe1.first_addition"},
                    },
                    "edges": [
                        ("first_addition", "double"),
                        ("double", "second_addition"),
                        ("second_addition", "third_addition"),
                    ],
                },
                "pipe2": {
                    "metadata": {"type": "another test pipeline", "author": "you"},
                    "max_loops_allowed": 100,
                    "nodes": {
                        "first_addition": {"refer_to": "pipe1.first_addition", "run_parameters": {"add": 4}},
                        "double": {"type": "Double", "init_parameters": {"input": "value", "output": "value"}},
                        "second_addition": {"refer_to": "pipe1.second_addition"},
                    },
                    "edges": [
                        ("first_addition", "double"),
                        ("double", "second_addition"),
                    ],
                },
            },
            "dependencies": ["test", "canals"],
        }
    )

    pipe1 = pipelines["pipe1"]
    assert pipe1.metadata == {"type": "test pipeline", "author": "me"}

    first_addition = pipe1.get_node("first_addition")
    assert type(first_addition["instance"]) == AddValue
    assert first_addition["instance"].add == 1
    assert first_addition["parameters"] == {"add": 6}

    second_addition = pipe1.get_node("second_addition")
    assert type(second_addition["instance"]) == AddValue
    assert second_addition["instance"].add == 1
    assert second_addition["parameters"] == {}
    assert second_addition["instance"] != first_addition["instance"]

    third_addition = pipe1.get_node("third_addition")
    assert type(third_addition["instance"]) == AddValue
    assert third_addition["instance"].add == 1
    assert third_addition["parameters"] == {}
    assert third_addition["instance"] == first_addition["instance"]

    double = pipe1.get_node("double")
    assert type(double["instance"]) == Double
    assert double["parameters"] == {}

    assert list(pipe1.graph.edges) == [
        ("first_addition", "double"),
        ("double", "second_addition"),
        ("second_addition", "third_addition"),
    ]

    pipe2 = pipelines["pipe2"]
    assert pipe2.metadata == {"type": "another test pipeline", "author": "you"}

    first_addition_2 = pipe2.get_node("first_addition")
    assert type(first_addition_2["instance"]) == AddValue
    assert first_addition_2["instance"].add == 1
    assert first_addition_2["parameters"] == {"add": 4}
    assert first_addition_2["instance"] == first_addition["instance"]

    second_addition_2 = pipe2.get_node("second_addition")
    assert type(second_addition_2["instance"]) == AddValue
    assert second_addition_2["instance"].add == 1
    assert second_addition_2["parameters"] == {}
    assert second_addition_2["instance"] != first_addition_2["instance"]
    assert second_addition_2["instance"] == second_addition["instance"]

    with pytest.raises(ValueError):
        pipe2.get_node("third_addition")

    double_2 = pipe2.get_node("double")
    assert type(double_2["instance"]) == Double
    assert double_2["parameters"] == {}
    assert double_2["instance"] != double["instance"]

    assert list(pipe2.graph.edges) == [
        ("first_addition", "double"),
        ("double", "second_addition"),
    ]
