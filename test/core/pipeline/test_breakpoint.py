# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack.core.pipeline.breakpoint import (
    _transform_json_structure,
    _validate_break_point,
    _validate_resume_state,
    load_state,
)


def test_transform_json_structure_unwraps_sender_value():
    data = {
        "key1": [{"sender": None, "value": "some value"}],
        "key2": [{"sender": "comp1", "value": 42}],
        "key3": "direct value",
    }

    result = _transform_json_structure(data)

    assert result == {"key1": "some value", "key2": 42, "key3": "direct value"}


def test_transform_json_structure_handles_nested_structures():
    data = {
        "key1": [{"sender": None, "value": "value1"}],
        "key2": {"nested": [{"sender": "comp1", "value": "value2"}], "direct": "value3"},
        "key3": [[{"sender": None, "value": "value4"}], [{"sender": "comp2", "value": "value5"}]],
    }

    result = _transform_json_structure(data)

    assert result == {"key1": "value1", "key2": {"nested": "value2", "direct": "value3"}, "key3": ["value4", "value5"]}


def test_validate_resume_state_validates_required_keys():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        # Missing pipeline_state
    }

    with pytest.raises(ValueError, match="Invalid state file: missing required keys"):
        _validate_resume_state(state)

    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {},
            # Missing ordered_component_names
        },
    }

    with pytest.raises(ValueError, match="Invalid pipeline_state: missing required keys"):
        _validate_resume_state(state)


def test_validate_resume_state_validates_component_consistency():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"],  # inconsistent with component_visits
        },
    }

    with pytest.raises(ValueError, match="Inconsistent state: components in pipeline_state"):
        _validate_resume_state(state)


def test_validate_resume_state_validates_valid_state():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
        },
    }

    _validate_resume_state(state)  # should not raise any exception


def test_load_state_loads_valid_state(tmp_path):
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
        },
    }
    state_file = tmp_path / "state.json"
    with open(state_file, "w") as f:
        json.dump(state, f)

    loaded_state = load_state(state_file)
    assert loaded_state == state


def test_load_state_handles_invalid_state(tmp_path):
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"],  # inconsistent with component_visits
        },
    }

    state_file = tmp_path / "invalid_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f)

    with pytest.raises(ValueError, match="Invalid pipeline state from"):
        load_state(state_file)
