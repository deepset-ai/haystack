# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import Breakpoint, ToolBreakpoint
from haystack.tools import Tool
from test.components.agents.test_agent import (
    MockChatGeneratorWithoutRunAsync,
    MockChatGeneratorWithRunAsync,
    weather_function,
)


def create_chat_generator_breakpoint(visit_count: int = 0) -> Breakpoint:
    return Breakpoint(component_name="chat_generator", visit_count=visit_count)


def create_tool_breakpoint(tool_name: Optional[str] = None, visit_count: int = 0) -> ToolBreakpoint:
    return ToolBreakpoint(component_name="tool_invoker", visit_count=visit_count, tool_name=tool_name)


# Common fixtures
@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        function=weather_function,
    )


@pytest.fixture
def debug_path(tmp_path):
    return str(tmp_path / "debug_states")


@pytest.fixture
def agent_sync(weather_tool):
    generator = MockChatGeneratorWithoutRunAsync()
    mock_run = MagicMock()
    mock_run.return_value = {
        "replies": [
            ChatMessage.from_assistant(
                "I'll help you check the weather.",
                tool_calls=[{"tool_name": "weather_tool", "tool_args": {"location": "Berlin"}}],
            )
        ]
    }

    def mock_run_with_tools(messages, tools=None, **kwargs):
        return mock_run.return_value

    generator.run = mock_run_with_tools

    return Agent(
        chat_generator=generator,
        tools=[weather_tool],
        system_prompt="You are a helpful assistant that can use tools to help users.",
    )


@pytest.fixture
def mock_agent_with_tool_calls_sync(monkeypatch, weather_tool):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = OpenAIChatGenerator()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=1)
    agent.warm_up()
    agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})
    return agent


@pytest.fixture
def agent_async(weather_tool):
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
    return Agent(
        chat_generator=generator,
        tools=[weather_tool],
        system_prompt="You are a helpful assistant that can use tools to help users.",
    )


@pytest.fixture
def mock_agent_with_tool_calls_async(monkeypatch, weather_tool):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = MockChatGeneratorWithRunAsync()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=1)
    agent.warm_up()
    agent.chat_generator.run_async = AsyncMock(return_value={"replies": mock_messages})
    return agent
