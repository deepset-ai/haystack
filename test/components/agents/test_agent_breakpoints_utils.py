# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from test.components.agents.test_agent import MockChatGenerator, MockChatGeneratorWithoutRunAsync, weather_function


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
    return str(tmp_path / "debug_snapshots")


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
    generator = MockChatGenerator()
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
    generator = MockChatGenerator()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=1)
    agent.warm_up()
    agent.chat_generator.run_async = AsyncMock(return_value={"replies": mock_messages})
    return agent
