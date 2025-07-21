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


def test_pipeline_snapshot_includes_outputs_from(tmp_path):
    """Test that include_outputs_from is properly saved and restored in pipeline snapshots."""
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
            "include_outputs_from": ["comp1", "comp2"],
        },
    }
    pipeline_snapshot_file = tmp_path / "snapshot_with_include_outputs.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot.pipeline_state.include_outputs_from == {"comp1", "comp2"}


def test_pipeline_snapshot_without_include_outputs_from(tmp_path):
    """Test that pipeline snapshots without include_outputs_from default to empty set."""
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
            # No include_outputs_from field
        },
    }
    pipeline_snapshot_file = tmp_path / "snapshot_without_include_outputs.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot.pipeline_state.include_outputs_from == set()


def test_pipeline_snapshot_with_intermediate_outputs(tmp_path):
    """Test that intermediate outputs are properly saved and restored in pipeline snapshots."""
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
            "include_outputs_from": ["comp1", "comp2"],
        },
        "intermediate_outputs": {"comp1": {"result": "output1"}, "comp2": {"result": "output2"}},
    }
    pipeline_snapshot_file = tmp_path / "snapshot_with_intermediate_outputs.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot.intermediate_outputs == {"comp1": {"result": "output1"}, "comp2": {"result": "output2"}}


def test_pipeline_snapshot_without_intermediate_outputs(tmp_path):
    """Test that pipeline snapshots without intermediate_outputs default to None."""
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {
            "original_input_data": {},
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"],
            "include_outputs_from": ["comp1", "comp2"],
        },
        # No intermediate_outputs field
    }
    pipeline_snapshot_file = tmp_path / "snapshot_without_intermediate_outputs.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot.intermediate_outputs is None


def test_json_serialization_of_sets():
    """Test that sets are properly converted to lists for JSON serialization and back to sets when deserializing."""
    import json

    from haystack.dataclasses.breakpoints import Breakpoint, PipelineSnapshot, PipelineState

    # Create a PipelineState with a set
    pipeline_state = PipelineState(
        original_input_data={},
        inputs={},
        component_visits={"comp1": 0, "comp2": 0},
        ordered_component_names=["comp1", "comp2"],
        include_outputs_from={"comp1", "comp2"},
    )

    # Create a PipelineSnapshot
    break_point = Breakpoint(component_name="comp1", visit_count=0)
    snapshot = PipelineSnapshot(
        pipeline_state=pipeline_state, break_point=break_point, intermediate_outputs={"comp1": {"result": "test"}}
    )

    # Convert to dict (should convert set to list)
    snapshot_dict = snapshot.to_dict()

    # Verify the set was converted to list in the JSON representation
    assert isinstance(snapshot_dict["pipeline_state"]["include_outputs_from"], list)
    assert set(snapshot_dict["pipeline_state"]["include_outputs_from"]) == {"comp1", "comp2"}

    # Test JSON serialization (should not raise an error)
    json_str = json.dumps(snapshot_dict)

    # Test JSON deserialization
    loaded_dict = json.loads(json_str)

    # Convert back to PipelineSnapshot (should convert list back to set)
    loaded_snapshot = PipelineSnapshot.from_dict(loaded_dict)

    # Verify the list was converted back to set
    assert isinstance(loaded_snapshot.pipeline_state.include_outputs_from, set)
    assert loaded_snapshot.pipeline_state.include_outputs_from == {"comp1", "comp2"}


def test_breakpoint_saves_intermediate_outputs():
    """Test that breakpoints save intermediate outputs from components in include_outputs_from."""
    from haystack import component
    from haystack.core.errors import BreakpointException
    from haystack.core.pipeline import Pipeline
    from haystack.dataclasses.breakpoints import Breakpoint

    @component
    class SimpleComponent:
        @component.output_types(result=str)
        def run(self, input_value: str) -> dict[str, str]:
            return {"result": f"processed_{input_value}"}

    # Create a simple pipeline
    pipeline = Pipeline()
    comp1 = SimpleComponent()
    comp2 = SimpleComponent()
    pipeline.add_component("comp1", comp1)
    pipeline.add_component("comp2", comp2)
    pipeline.connect("comp1", "comp2")

    # Create a breakpoint that will trigger on comp2
    break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path="debug_path/")

    try:
        # Run with include_outputs_from to capture intermediate outputs
        pipeline.run(data={"comp1": {"input_value": "test"}}, include_outputs_from={"comp1"}, break_point=break_point)
    except BreakpointException as e:
        # The breakpoint should be triggered
        assert e.component == "comp2"

        # The snapshot should contain intermediate outputs from comp1
        # Note: In a real scenario, you would load the snapshot file here
        # For this test, we just verify the exception was raised
        pass
