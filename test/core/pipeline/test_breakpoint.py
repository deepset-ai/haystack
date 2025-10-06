# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack import component
from haystack.core.errors import BreakpointException
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.breakpoint import (
    _transform_json_structure,
    _trigger_chat_generator_breakpoint,
    _trigger_tool_invoker_breakpoint,
    load_pipeline_snapshot,
)
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import (
    AgentBreakpoint,
    AgentSnapshot,
    Breakpoint,
    PipelineSnapshot,
    PipelineState,
    ToolBreakpoint,
)


@pytest.fixture
def make_pipeline_snapshot():
    def _make(break_point: AgentBreakpoint) -> PipelineSnapshot:
        return PipelineSnapshot(
            break_point=break_point,
            pipeline_state=PipelineState(inputs={}, component_visits={"agent": 0}, pipeline_outputs={}),
            original_input_data={},
            ordered_component_names=["agent"],
            agent_snapshot=AgentSnapshot(
                break_point=break_point,
                component_inputs={"chat_generator": {}, "tool_invoker": {"serialized_data": {"state": {}}}},
                component_visits={"chat_generator": 0, "tool_invoker": 0},
            ),
        )

    return _make


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
        "pipeline_state": {"inputs": {}, "component_visits": {"comp1": 0, "comp2": 0}, "pipeline_outputs": {}},
        "original_input_data": {},
        "ordered_component_names": ["comp1", "comp2"],
        "include_outputs_from": ["comp1", "comp2"],
    }
    pipeline_snapshot_file = tmp_path / "state.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    loaded_snapshot = load_pipeline_snapshot(pipeline_snapshot_file)
    assert loaded_snapshot == PipelineSnapshot.from_dict(pipeline_snapshot)


def test_load_state_handles_invalid_state(tmp_path):
    pipeline_snapshot = {
        "break_point": {"component_name": "comp1", "visit_count": 0},
        "pipeline_state": {"inputs": {}, "component_visits": {"comp1": 0, "comp2": 0}, "pipeline_outputs": {}},
        "original_input_data": {},
        "include_outputs_from": ["comp1", "comp2"],
        "ordered_component_names": ["comp1", "comp3"],  # inconsistent with component_visits
    }

    pipeline_snapshot_file = tmp_path / "invalid_pipeline_snapshot.json"
    with open(pipeline_snapshot_file, "w") as f:
        json.dump(pipeline_snapshot, f)

    with pytest.raises(ValueError, match="Invalid pipeline snapshot from"):
        load_pipeline_snapshot(pipeline_snapshot_file)


def test_breakpoint_saves_intermediate_outputs(tmp_path):
    @component
    class SimpleComponent:
        @component.output_types(result=str)
        def run(self, input_value: str) -> dict[str, str]:
            return {"result": f"processed_{input_value}"}

    pipeline = Pipeline()
    comp1 = SimpleComponent()
    comp2 = SimpleComponent()
    pipeline.add_component("comp1", comp1)
    pipeline.add_component("comp2", comp2)
    pipeline.connect("comp1", "comp2")

    # breakpoint on comp2
    break_point = Breakpoint(component_name="comp2", visit_count=0, snapshot_file_path=str(tmp_path))

    try:
        # run with include_outputs_from to capture intermediate outputs
        pipeline.run(data={"comp1": {"input_value": "test"}}, include_outputs_from={"comp1"}, break_point=break_point)
    except BreakpointException as e:
        # breakpoint should be triggered
        assert e.component == "comp2"

        # verify snapshot file contains the intermediate outputs
        snapshot_files = list(tmp_path.glob("comp2_*.json"))
        assert len(snapshot_files) == 1, f"Expected exactly one snapshot file, found {len(snapshot_files)}"

        snapshot_file = snapshot_files[0]
        loaded_snapshot = load_pipeline_snapshot(snapshot_file)

        # verify the snapshot contains the intermediate outputs from comp1
        assert "comp1" in loaded_snapshot.pipeline_state.pipeline_outputs
        assert loaded_snapshot.pipeline_state.pipeline_outputs["comp1"]["result"] == "processed_test"

        # verify the whole pipeline state contains the expected data
        assert loaded_snapshot.pipeline_state.component_visits["comp1"] == 1
        assert loaded_snapshot.pipeline_state.component_visits["comp2"] == 0
        assert "comp1" in loaded_snapshot.include_outputs_from
        assert loaded_snapshot.break_point.component_name == "comp2"
        assert loaded_snapshot.break_point.visit_count == 0


def test_trigger_tool_invoker_breakpoint(make_pipeline_snapshot):
    pipeline_snapshot_with_agent_breakpoint = make_pipeline_snapshot(
        break_point=AgentBreakpoint("agent", ToolBreakpoint(component_name="tool_invoker"))
    )
    with pytest.raises(BreakpointException):
        _trigger_tool_invoker_breakpoint(
            llm_messages=[ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="tool1", arguments={})])],
            pipeline_snapshot=pipeline_snapshot_with_agent_breakpoint,
        )


def test_trigger_tool_invoker_breakpoint_no_raise(make_pipeline_snapshot):
    pipeline_snapshot_with_agent_breakpoint = make_pipeline_snapshot(
        break_point=AgentBreakpoint("agent", ToolBreakpoint(component_name="tool_invoker", tool_name="tool2"))
    )
    # This should not raise since the tool call is for "tool1", not "tool2"
    _trigger_tool_invoker_breakpoint(
        llm_messages=[ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="tool1", arguments={})])],
        pipeline_snapshot=pipeline_snapshot_with_agent_breakpoint,
    )


def test_trigger_tool_invoker_breakpoint_specific_tool(make_pipeline_snapshot):
    """
    This is to test if a specific tool is set in the ToolBreakpoint, the BreakpointException is raised even when
    there are multiple tool calls in the message.
    """
    pipeline_snapshot_with_agent_breakpoint = make_pipeline_snapshot(
        break_point=AgentBreakpoint("agent", ToolBreakpoint(component_name="tool_invoker", tool_name="tool2"))
    )
    with pytest.raises(BreakpointException):
        _trigger_tool_invoker_breakpoint(
            llm_messages=[
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="tool1", arguments={}), ToolCall(tool_name="tool2", arguments={})]
                )
            ],
            pipeline_snapshot=pipeline_snapshot_with_agent_breakpoint,
        )


def test_trigger_chat_generator_breakpoint(make_pipeline_snapshot):
    pipeline_snapshot_with_agent_breakpoint = make_pipeline_snapshot(
        break_point=AgentBreakpoint("agent", Breakpoint(component_name="chat_generator"))
    )
    with pytest.raises(BreakpointException):
        _trigger_chat_generator_breakpoint(pipeline_snapshot=pipeline_snapshot_with_agent_breakpoint)
