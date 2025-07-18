# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack.core.pipeline.breakpoint import _transform_json_structure, load_pipeline_snapshot
from haystack.dataclasses.breakpoints import PipelineSnapshot


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


def test_load_pipeline_snapshot_loads_valid_snapshot(tmp_path):
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
        },
    }
    pipeline_snapshot_file = tmp_path / "state.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot == PipelineSnapshot.from_dict(pipeline_snapshot)


def test_load_state_handles_invalid_state(tmp_path):
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"],  # inconsistent with component_visits
        },
    }

    pipeline_snapshot_file = tmp_path / "invalid_pipeline_snapshot.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    with pytest.raises(ValueError, match="Invalid pipeline snapshot from"):
        load_pipeline_snapshot(pipeline_snapshot_file)
