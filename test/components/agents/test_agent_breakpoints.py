# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, Breakpoint, ToolBreakpoint
from haystack.tools import Tool
from test.components.agents.test_agent import MockChatGenerator, weather_function


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        function=weather_function,
    )


@pytest.fixture
def mock_chat_generator():
    generator = MockChatGenerator()
    reply = {
        "replies": [
            ChatMessage.from_assistant(
                "I'll help you check the weather.",
                tool_calls=[{"tool_name": "weather_tool", "tool_args": {"location": "Berlin"}}],
            )
        ]
    }

    def run(messages, tools=None, **kwargs):
        return reply

    async def run_async(messages, tools=None, **kwargs):
        return reply

    generator.run = run
    generator.run_async = run_async
    return generator


@pytest.fixture
def agent(mock_chat_generator, weather_tool):
    return Agent(
        chat_generator=mock_chat_generator,
        tools=[weather_tool],
        system_prompt="You are a helpful assistant that can use tools to help users.",
        max_agent_steps=10,  # Increase max steps to allow breakpoints to trigger
    )


@pytest.fixture
def mock_agent_with_tool_calls(monkeypatch, weather_tool):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = MockChatGenerator()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=10)  # Increase max steps
    agent.warm_up()
    agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})
    agent.chat_generator.run_async = AsyncMock(return_value={"replies": mock_messages})
    return agent


class TestAgentBreakpoints:
    def test_run_with_chat_generator_breakpoint(self, agent):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        chat_generator_bp = Breakpoint(component_name="chat_generator", visit_count=0)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")
        with pytest.raises(BreakpointException) as exc_info:
            agent.run(messages=messages, break_point=agent_breakpoint)
        assert exc_info.value.component == "chat_generator"
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    def test_run_with_tool_invoker_breakpoint(self, mock_agent_with_tool_calls):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
        with pytest.raises(BreakpointException) as exc_info:
            mock_agent_with_tool_calls.run(messages=messages, break_point=agent_breakpoint)

        assert exc_info.value.component == "tool_invoker"
        assert {"chat_generator", "tool_invoker"} == set(exc_info.value.inputs.keys())
        assert "serialization_schema" in exc_info.value.inputs["chat_generator"]
        assert "serialized_data" in exc_info.value.inputs["chat_generator"]
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    def test_resume_from_chat_generator(self, agent, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        chat_generator_bp = Breakpoint(component_name="chat_generator", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")

        try:
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_chat_generator_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = agent.run(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    def test_resume_from_tool_invoker(self, mock_agent_with_tool_calls, tmp_path):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        debug_path = str(tmp_path / "debug_snapshots")
        tool_bp = ToolBreakpoint(component_name="tool_invoker", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")

        try:
            mock_agent_with_tool_calls.run(messages=messages, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = mock_agent_with_tool_calls.run(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    def test_invalid_combination_breakpoint_and_pipeline_snapshot(self, mock_agent_with_tool_calls):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
        with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
            mock_agent_with_tool_calls.run(
                messages=messages,
                break_point=agent_breakpoint,
                snapshot=AgentSnapshot(component_inputs={}, component_visits={}, break_point=agent_breakpoint),
            )

    def test_breakpoint_with_invalid_component(self, mock_agent_with_tool_calls):
        invalid_bp = Breakpoint(component_name="invalid_breakpoint")
        with pytest.raises(ValueError):
            AgentBreakpoint(break_point=invalid_bp, agent_name="test_agent")

    def test_breakpoint_with_invalid_tool_name(self, mock_agent_with_tool_calls):
        tool_breakpoint = ToolBreakpoint(component_name="tool_invoker", tool_name="invalid_tool")
        with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
            agent_breakpoints = AgentBreakpoint(break_point=tool_breakpoint, agent_name="test_agent")
            mock_agent_with_tool_calls.run(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoints
            )


class TestAsyncAgentBreakpoints:
    @pytest.mark.asyncio
    async def test_run_async_with_chat_generator_breakpoint(self, agent):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        chat_generator_bp = Breakpoint(component_name="chat_generator")
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")
        with pytest.raises(BreakpointException) as exc_info:
            await agent.run_async(messages=messages, break_point=agent_breakpoint)
        assert exc_info.value.component == "chat_generator"
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    @pytest.mark.asyncio
    async def test_run_async_with_tool_invoker_breakpoint(self, mock_agent_with_tool_calls):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test")
        with pytest.raises(BreakpointException) as exc_info:
            await mock_agent_with_tool_calls.run_async(messages=messages, break_point=agent_breakpoint)

        assert exc_info.value.component == "tool_invoker"
        assert "messages" in exc_info.value.inputs["tool_invoker"]["serialized_data"]

    @pytest.mark.asyncio
    async def test_resume_from_chat_generator_async(self, agent, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        chat_generator_bp = Breakpoint(component_name="chat_generator", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")

        try:
            await agent.run_async(messages=messages, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_chat_generator_*.json"))

        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = await agent.run_async(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_resume_from_tool_invoker_async(self, mock_agent_with_tool_calls, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")

        try:
            await mock_agent_with_tool_calls.run_async(messages=messages, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))

        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = await mock_agent_with_tool_calls.run_async(
            messages=[ChatMessage.from_user("Continue from where we left off.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_combination_breakpoint_and_pipeline_snapshot_async(self, mock_agent_with_tool_calls):
        tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test")
        with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
            await mock_agent_with_tool_calls.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")],
                break_point=agent_breakpoint,
                snapshot=AgentSnapshot(component_inputs={}, component_visits={}, break_point=agent_breakpoint),
            )

    @pytest.mark.asyncio
    async def test_breakpoint_with_invalid_component_async(self, mock_agent_with_tool_calls):
        with pytest.raises(ValueError):
            AgentBreakpoint(break_point=Breakpoint(component_name="invalid_breakpoint"), agent_name="test")

    @pytest.mark.asyncio
    async def test_breakpoint_with_invalid_tool_name_async(self, mock_agent_with_tool_calls):
        tool_breakpoint = ToolBreakpoint(component_name="tool_invoker", tool_name="invalid_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_breakpoint, agent_name="test")
        with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
            await mock_agent_with_tool_calls.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
            )
