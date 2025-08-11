# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from haystack.components.agents import Agent
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.tools import Tool
from test.components.agents.test_agent import MockChatGeneratorWithRunAsync, weather_function

AGENT_NAME = "isolated_agent"


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
    generator = MockChatGeneratorWithRunAsync()
    mock_run_async = AsyncMock()
    mock_run_async.return_value = {
        "replies": [
            ChatMessage.from_assistant(
                "I'll help you check the weather.",
                tool_calls=[{"tool_name": "weather_tool", "tool_args": {"location": "Berlin"}}],
            )
        ]
    }

    async def mock_run_async_with_tools(messages, tools=None, **kwargs):
        return mock_run_async.return_value

    generator.run_async = mock_run_async_with_tools
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
def debug_path(tmp_path):
    return str(tmp_path / "debug_snapshots")


@pytest.fixture
def mock_agent_with_tool_calls(monkeypatch, weather_tool):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = MockChatGeneratorWithRunAsync()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=10)  # Increase max steps
    agent.warm_up()
    agent.chat_generator.run_async = AsyncMock(return_value={"replies": mock_messages})
    return agent


@pytest.mark.asyncio
async def test_run_async_with_chat_generator_breakpoint(agent):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = Breakpoint(component_name="chat_generator", visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test")
    with pytest.raises(BreakpointException) as exc_info:
        await agent.run_async(messages=messages, break_point=agent_breakpoint, agent_name=AGENT_NAME)
    assert exc_info.value.component == "chat_generator"
    assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]


@pytest.mark.asyncio
async def test_run_async_with_tool_invoker_breakpoint(mock_agent_with_tool_calls):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test")
    with pytest.raises(BreakpointException) as exc_info:
        await mock_agent_with_tool_calls.run_async(
            messages=messages, break_point=agent_breakpoint, agent_name=AGENT_NAME
        )

    assert exc_info.value.component == "tool_invoker"
    assert "messages" in exc_info.value.inputs["tool_invoker"]["serialized_data"]


@pytest.mark.asyncio
async def test_resume_from_chat_generator_async(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = Breakpoint(component_name="chat_generator", visit_count=0, snapshot_file_path=debug_path)
    agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name=AGENT_NAME)

    try:
        await agent.run_async(messages=messages, break_point=agent_breakpoint, agent_name=AGENT_NAME)
    except BreakpointException:
        pass

    snapshot_files = list(Path(debug_path).glob(AGENT_NAME + "_chat_generator_*.json"))

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
async def test_resume_from_tool_invoker_async(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = ToolBreakpoint(
        component_name="tool_invoker", visit_count=0, tool_name="weather_tool", snapshot_file_path=debug_path
    )
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name=AGENT_NAME)

    try:
        await mock_agent_with_tool_calls.run_async(
            messages=messages, break_point=agent_breakpoint, agent_name=AGENT_NAME
        )
    except BreakpointException:
        pass

    snapshot_files = list(Path(debug_path).glob(AGENT_NAME + "_tool_invoker_*.json"))

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
async def test_invalid_combination_breakpoint_and_pipeline_snapshot_async(mock_agent_with_tool_calls):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test")
    with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
        await mock_agent_with_tool_calls.run_async(
            messages=messages, break_point=agent_breakpoint, snapshot={"some": "snapshot"}
        )


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_component_async(mock_agent_with_tool_calls):
    invalid_bp = Breakpoint(component_name="invalid_breakpoint", visit_count=0)
    with pytest.raises(ValueError):
        AgentBreakpoint(break_point=invalid_bp, agent_name="test")


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_tool_name_async(mock_agent_with_tool_calls):
    tool_breakpoint = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="invalid_tool")
    with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
        agent_breakpoint = AgentBreakpoint(break_point=tool_breakpoint, agent_name="test")
        await mock_agent_with_tool_calls.run_async(
            messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
        )
