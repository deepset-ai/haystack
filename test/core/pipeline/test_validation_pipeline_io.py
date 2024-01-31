# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import pytest

from haystack.core.component.types import InputSocket, OutputSocket, Variadic
from haystack.core.errors import PipelineValidationError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.descriptions import find_pipeline_inputs, find_pipeline_outputs
from haystack.testing.sample_components import AddFixedValue, Double, Parity, Sum


def test_find_pipeline_input_no_input():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    assert find_pipeline_inputs(pipe.graph) == {"comp1": [], "comp2": []}


def test_find_pipeline_input_one_input():
    pipe = Pipeline()
    comp1 = Double()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [InputSocket(name="value", type=int, component=comp1)],
        "comp2": [],
    }


def test_find_pipeline_input_two_inputs_same_component():
    pipe = Pipeline()
    comp1 = AddFixedValue()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, component=comp1),
            InputSocket(name="add", type=Optional[int], default_value=None, component=comp1),
        ],
        "comp2": [],
    }


def test_find_pipeline_input_some_inputs_different_components():
    pipe = Pipeline()
    comp1 = AddFixedValue()
    comp2 = Double()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.add_component("comp3", AddFixedValue())
    pipe.connect("comp1.result", "comp3.value")
    pipe.connect("comp2.value", "comp3.add")
    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, component=comp1),
            InputSocket(name="add", type=Optional[int], default_value=None, component=comp1),
        ],
        "comp2": [InputSocket(name="value", type=int, component=comp2)],
        "comp3": [],
    }


def test_find_pipeline_variable_input_nodes_in_the_pipeline():
    pipe = Pipeline()
    comp1 = AddFixedValue()
    comp2 = Double()
    comp3 = Sum()
    pipe.add_component("comp1", comp1)
    pipe.add_component("comp2", comp2)
    pipe.add_component("comp3", comp3)

    assert find_pipeline_inputs(pipe.graph) == {
        "comp1": [
            InputSocket(name="value", type=int, component=comp1),
            InputSocket(name="add", type=Optional[int], default_value=None, component=comp1),
        ],
        "comp2": [InputSocket(name="value", type=int, component=comp2)],
        "comp3": [InputSocket(name="values", type=Variadic[int], component=comp3)],
    }


def test_find_pipeline_output_no_output():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")

    assert find_pipeline_outputs(pipe.graph) == {"comp1": [], "comp2": []}


def test_find_pipeline_output_one_output():
    pipe = Pipeline()
    comp2 = Double()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2")

    assert find_pipeline_outputs(pipe.graph) == {
        "comp1": [],
        "comp2": [OutputSocket(name="value", type=int, component=comp2)],
    }


def test_find_pipeline_some_outputs_same_component():
    pipe = Pipeline()
    comp2 = Parity()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", comp2)
    pipe.connect("comp1", "comp2")

    assert find_pipeline_outputs(pipe.graph) == {
        "comp1": [],
        "comp2": [
            OutputSocket(name="even", type=int, component=comp2),
            OutputSocket(name="odd", type=int, component=comp2),
        ],
    }


def test_find_pipeline_some_outputs_different_components():
    pipe = Pipeline()
    comp2 = Parity()
    comp3 = Double()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", comp2)
    pipe.add_component("comp3", comp3)
    pipe.connect("comp1", "comp2")
    pipe.connect("comp1", "comp3")

    assert find_pipeline_outputs(pipe.graph) == {
        "comp1": [],
        "comp2": [
            OutputSocket(name="even", type=int, component=comp2),
            OutputSocket(name="odd", type=int, component=comp2),
        ],
        "comp3": [OutputSocket(name="value", type=int, component=comp3)],
    }


def test_validate_pipeline_input_pipeline_with_no_inputs():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.connect("comp2", "comp1")
    res = pipe.run({})
    assert res == {}


def test_validate_pipeline_input_unknown_component():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    with pytest.raises(ValueError):
        pipe.run({"test_component": {"value": 1}})


def test_validate_pipeline_input_all_necessary_input_is_present():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    with pytest.raises(ValueError):
        pipe.run({})


def test_validate_pipeline_input_all_necessary_input_is_present_considering_defaults():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    pipe.run({"comp1": {"value": 1}})
    pipe.run({"comp1": {"value": 1, "add": 2}})
    with pytest.raises(ValueError):
        pipe.run({"comp1": {"add": 3}})


def test_validate_pipeline_input_only_expected_input_is_present():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    with pytest.raises(ValueError):
        pipe.run({"comp1": {"value": 1}, "comp2": {"value": 2}})


def test_validate_pipeline_input_only_expected_input_is_present_falsy():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    with pytest.raises(ValueError):
        pipe.run({"comp1": {"value": 1}, "comp2": {"value": 0}})


def test_validate_pipeline_falsy_input_present():
    pipe = Pipeline()
    pipe.add_component("comp", Double())
    assert pipe.run({"comp": {"value": 0}}) == {"comp": {"value": 0}}


def test_validate_pipeline_falsy_input_missing():
    pipe = Pipeline()
    pipe.add_component("comp", Double())
    with pytest.raises(ValueError):
        pipe.run({"comp": {}})


def test_validate_pipeline_input_only_expected_input_is_present_including_unknown_names():
    pipe = Pipeline()
    pipe.add_component("comp1", Double())
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")

    with pytest.raises(ValueError):
        pipe.run({"comp1": {"value": 1, "add": 2}})


def test_validate_pipeline_input_only_expected_input_is_present_and_defaults_dont_interfere():
    pipe = Pipeline()
    pipe.add_component("comp1", AddFixedValue(add=10))
    pipe.add_component("comp2", Double())
    pipe.connect("comp1", "comp2")
    assert pipe.run({"comp1": {"value": 1, "add": 5}}) == {"comp2": {"value": 12}}
