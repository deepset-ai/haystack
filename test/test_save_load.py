from pathlib import Path

from canals import Pipeline
from canals.pipeline._utils import marshal_pipelines, save_pipelines
from test.nodes import AddValue, Double

import logging

logging.basicConfig(level=logging.DEBUG)


def test_save():
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
                "edges": [
                    ("first_addition", "double"),
                    ("double", "second_addition"),
                    ("second_addition", "third_addition"),
                ],
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
                    "third_addition": {"refer_to": "first_addition"},
                },
            },
            "pipe2": {
                "metadata": {"type": "another test pipeline", "author": "you"},
                "max_loops_allowed": 100,
                "edges": [
                    ("first_addition", "double"),
                    ("double", "second_addition"),
                ],
                "nodes": {
                    "first_addition": {"refer_to": "pipe1.first_addition", "run_parameters": {"add": 4}},
                    "double": {"type": "Double", "init_parameters": {"input": "value", "output": "value"}},
                    "second_addition": {"refer_to": "pipe1.second_addition"},
                },
            },
        },
        "dependencies": ["test", "canals"],
    }
