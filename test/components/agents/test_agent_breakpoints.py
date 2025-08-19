# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from test.components.agents.test_agent_breakpoints_utils import agent_sync, mock_agent_with_tool_calls_sync


class TestAgentBreakpoints:
    def test_run_with_chat_generator_breakpoint(self, agent_sync):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        chat_generator_bp = Breakpoint(component_name="chat_generator", visit_count=0)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")
        with pytest.raises(BreakpointException) as exc_info:
            agent_sync.run(messages=messages, break_point=agent_breakpoint, agent_name="test")
        assert exc_info.value.component == "chat_generator"
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    def test_run_with_tool_invoker_breakpoint(self, mock_agent_with_tool_calls_sync):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
        with pytest.raises(BreakpointException) as exc_info:
            mock_agent_with_tool_calls_sync.run(messages=messages, break_point=agent_breakpoint, agent_name="test")

        assert exc_info.value.component == "tool_invoker"
        assert {"chat_generator", "tool_invoker"} == set(exc_info.value.inputs.keys())
        assert "serialization_schema" in exc_info.value.inputs["chat_generator"]
        assert "serialized_data" in exc_info.value.inputs["chat_generator"]
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    def test_resume_from_chat_generator(self, agent_sync, tmp_path):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        debug_path = str(tmp_path / "debug_snapshots")
        chat_generator_bp = Breakpoint(component_name="chat_generator", visit_count=0, snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")

        try:
            agent_sync.run(messages=messages, break_point=agent_breakpoint, agent_name="test_agent")
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_chat_generator_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = agent_sync.run(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    def test_resume_from_tool_invoker(self, mock_agent_with_tool_calls_sync, tmp_path):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        debug_path = str(tmp_path / "debug_snapshots")
        tool_bp = ToolBreakpoint(
            component_name="tool_invoker", visit_count=0, tool_name=None, snapshot_file_path=debug_path
        )
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")

        try:
            mock_agent_with_tool_calls_sync.run(
                messages=messages, break_point=agent_breakpoint, agent_name="test_agent"
            )
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = mock_agent_with_tool_calls_sync.run(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    def test_invalid_combination_breakpoint_and_pipeline_snapshot(self, mock_agent_with_tool_calls_sync):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
        with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
            mock_agent_with_tool_calls_sync.run(
                messages=messages, break_point=agent_breakpoint, snapshot={"some": "snapshot"}
            )

    def test_breakpoint_with_invalid_component(self, mock_agent_with_tool_calls_sync):
        invalid_bp = Breakpoint(component_name="invalid_breakpoint", visit_count=0)
        with pytest.raises(ValueError):
            AgentBreakpoint(break_point=invalid_bp, agent_name="test_agent")

    def test_breakpoint_with_invalid_tool_name(self, mock_agent_with_tool_calls_sync):
        tool_breakpoint = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="invalid_tool")
        with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
            agent_breakpoints = AgentBreakpoint(break_point=tool_breakpoint, agent_name="test_agent")
            mock_agent_with_tool_calls_sync.run(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoints
            )
