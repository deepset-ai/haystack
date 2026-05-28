# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jinja2 import TemplateSyntaxError
from openai import Stream
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk

from haystack import Document, Pipeline, component, tracing
from haystack.components.agents.agent import Agent
from haystack.components.agents.state import merge_lists
from haystack.components.agents.tool_calling import _run_tool
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners.branch import BranchJoiner
from haystack.components.joiners.list_joiner import ListJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.core.component.types import OutputSocket
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.chat_message import ChatRole, TextContent
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool, Tool, tool
from haystack.tools.toolset import Toolset
from haystack.tracing.logging_tracer import LoggingTracer
from haystack.utils import Secret, serialize_callable


def _user_msg(text: str) -> str:
    return f'{{% message role="user" %}}{text}{{% endmessage %}}'


def _sys_msg(text: str) -> str:
    return f'{{% message role="system" %}}{text}{{% endmessage %}}'


def sync_streaming_callback(chunk: StreamingChunk) -> None:
    """A synchronous streaming callback."""
    pass


async def async_streaming_callback(chunk: StreamingChunk) -> None:
    """An asynchronous streaming callback."""
    pass


def weather_function(location):
    weather_info = {
        "berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    for city, result in weather_info.items():
        if city in location.lower():
            return result
    return {"weather": "unknown", "temperature": 0, "unit": "celsius"}


@tool
def weather_tool_with_decorator(location: str) -> str:
    """Provides weather information for a given location."""
    return f"Weather report for {location}: 20°C, sunny"


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        function=weather_function,
    )


@pytest.fixture
def component_tool():
    return ComponentTool(name="parrot", description="This is a parrot.", component=PromptBuilder(template="{{parrot}}"))


@pytest.fixture
def make_agent(weather_tool):
    def _factory(**kwargs):
        return Agent(chat_generator=MockChatGenerator(), tools=[weather_tool], **kwargs)

    return _factory


class OpenAIMockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)  # noqa: B026
        self.mock_chunk = mock_chunk

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        yield self.mock_chunk


@pytest.fixture
def openai_mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage=None,
        )
        mock_chat_completion_create.return_value = OpenAIMockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


@component
class MockChatGeneratorWithoutTools:
    """A mock chat generator that implements ChatGenerator protocol but doesn't support tools."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutTools", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGeneratorWithoutTools":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


@component
class MockChatGeneratorWithoutRunAsync:
    """A mock chat generator that implements ChatGenerator protocol but doesn't have run_async method."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutRunAsync", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGeneratorWithoutRunAsync":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


@component
class MockChatGenerator:
    def to_dict(self) -> dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutRunAsync", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGenerator":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello from run_async")]}


class TestAgent:
    def test_output_types(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool, component_tool])
        assert agent.__haystack_output__._sockets_dict == {
            "messages": OutputSocket(name="messages", type=list[ChatMessage], receivers=[]),
            "last_message": OutputSocket(name="last_message", type=ChatMessage, receivers=[]),
        }

    def test_to_dict(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
            exit_conditions=["text", "weather_tool"],
            state_schema={"foo": {"type": str}},
            tool_concurrency_limit=5,
            tool_streaming_callback_passthrough=True,
        )
        serialized_agent = agent.to_dict()
        # Verify the model is truthy and serialized
        assert "model" in serialized_agent["init_parameters"]["chat_generator"]["init_parameters"]
        model_name = serialized_agent["init_parameters"]["chat_generator"]["init_parameters"]["model"]
        # Check the rest of the structure
        expected_structure = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": model_name,
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "weather_tool",
                            "description": "Provides weather information for a given location.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                            "function": "test_agent.weather_function",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                    {
                        "type": "haystack.tools.component_tool.ComponentTool",
                        "data": {
                            "component": {
                                "type": "haystack.components.builders.prompt_builder.PromptBuilder",
                                "init_parameters": {
                                    "template": "{{parrot}}",
                                    "variables": None,
                                    "required_variables": "*",
                                },
                            },
                            "name": "parrot",
                            "description": "This is a parrot.",
                            "parameters": None,
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                ],
                "system_prompt": None,
                "user_prompt": None,
                "required_variables": None,
                "exit_conditions": ["text", "weather_tool"],
                "state_schema": {"foo": {"type": "str"}},
                "max_agent_steps": 100,
                "streaming_callback": None,
                "raise_on_tool_invocation_failure": False,
                "tool_concurrency_limit": 5,
                "tool_streaming_callback_passthrough": True,
                "confirmation_strategies": None,
            },
        }
        assert serialized_agent == expected_structure

    def test_to_dict_with_toolset(self, monkeypatch, weather_tool):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        toolset = Toolset(tools=[weather_tool])
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)
        serialized_agent = agent.to_dict()
        # Verify the model is truthy and serialized
        assert "model" in serialized_agent["init_parameters"]["chat_generator"]["init_parameters"]
        model_name = serialized_agent["init_parameters"]["chat_generator"]["init_parameters"]["model"]
        # Check the rest of the structure
        expected_structure = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": model_name,
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": {
                    "type": "haystack.tools.toolset.Toolset",
                    "data": {
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather_tool",
                                    "description": "Provides weather information for a given location.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"location": {"type": "string"}},
                                        "required": ["location"],
                                    },
                                    "function": "test_agent.weather_function",
                                    "outputs_to_string": None,
                                    "inputs_from_state": None,
                                    "outputs_to_state": None,
                                },
                            }
                        ]
                    },
                },
                "system_prompt": None,
                "user_prompt": None,
                "required_variables": None,
                "exit_conditions": ["text"],
                "state_schema": {},
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
                "tool_concurrency_limit": 4,
                "tool_streaming_callback_passthrough": False,
                "confirmation_strategies": None,
            },
        }
        assert serialized_agent == expected_structure

    def test_agent_serialization_with_tool_decorator(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[weather_tool_with_decorator])
        serialized_agent = agent.to_dict()
        deserialized_agent = Agent.from_dict(serialized_agent)

        assert deserialized_agent.tools == agent.tools
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        # Model name should match whatever the default is - not testing specific model
        assert deserialized_agent.chat_generator.model == agent.chat_generator.model
        assert deserialized_agent.chat_generator.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert deserialized_agent.exit_conditions == ["text"]

    def test_from_dict(self, monkeypatch):
        model = "gpt-5"
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        data = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": model,
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "weather_tool",
                            "description": "Provides weather information for a given location.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                            "function": "test_agent.weather_function",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                    {
                        "type": "haystack.tools.component_tool.ComponentTool",
                        "data": {
                            "component": {
                                "type": "haystack.components.builders.prompt_builder.PromptBuilder",
                                "init_parameters": {
                                    "template": "{{parrot}}",
                                    "variables": None,
                                    "required_variables": "*",
                                },
                            },
                            "name": "parrot",
                            "description": "This is a parrot.",
                            "parameters": None,
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                ],
                "system_prompt": None,
                "exit_conditions": ["text", "weather_tool"],
                "state_schema": {"foo": {"type": "str"}},
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
                "tool_concurrency_limit": 5,
                "tool_streaming_callback_passthrough": True,
            },
        }
        agent = Agent.from_dict(data)
        assert isinstance(agent, Agent)
        assert isinstance(agent.chat_generator, OpenAIChatGenerator)
        # from_dict should restore the model from the dict (testing backward compatibility)
        assert agent.chat_generator.model == model
        assert agent.chat_generator.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert agent.tools[0].function is weather_function
        assert isinstance(agent.tools[1]._component, PromptBuilder)
        assert agent.exit_conditions == ["text", "weather_tool"]
        assert agent.state_schema == {
            "foo": {"type": str},
            "messages": {"handler": merge_lists, "type": list[ChatMessage]},
        }
        assert agent.tool_concurrency_limit == 5
        assert agent.tool_streaming_callback_passthrough is True

    def test_from_dict_with_toolset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        data = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": {
                    "type": "haystack.tools.toolset.Toolset",
                    "data": {
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather_tool",
                                    "description": "Provides weather information for a given location.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"location": {"type": "string"}},
                                        "required": ["location"],
                                    },
                                    "function": "test_agent.weather_function",
                                    "outputs_to_string": None,
                                    "inputs_from_state": None,
                                    "outputs_to_state": None,
                                },
                            }
                        ]
                    },
                },
                "system_prompt": None,
                "exit_conditions": ["text"],
                "state_schema": {},
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
                "tool_concurrency_limit": 1,
                "tool_streaming_callback_passthrough": False,
            },
        }
        agent = Agent.from_dict(data)
        assert isinstance(agent, Agent)
        assert isinstance(agent.chat_generator, OpenAIChatGenerator)
        # from_dict should restore the model from the dict (testing backward compatibility)
        assert agent.chat_generator.model == "gpt-4o-mini"
        assert agent.chat_generator.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert isinstance(agent.tools, Toolset)
        assert agent.tools[0].function is weather_function
        assert agent.exit_conditions == ["text"]

    def test_from_dict_state_schema_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        data = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "weather_tool",
                            "description": "Provides weather information for a given location.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                            "function": "test_agent.weather_function",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                    {
                        "type": "haystack.tools.component_tool.ComponentTool",
                        "data": {
                            "component": {
                                "type": "haystack.components.builders.prompt_builder.PromptBuilder",
                                "init_parameters": {
                                    "template": "{{parrot}}",
                                    "variables": None,
                                    "required_variables": "*",
                                },
                            },
                            "name": "parrot",
                            "description": "This is a parrot.",
                            "parameters": None,
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    },
                ],
                "system_prompt": None,
                "exit_conditions": ["text", "weather_tool"],
                "state_schema": None,
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
                "tool_concurrency_limit": 5,
                "tool_streaming_callback_passthrough": True,
            },
        }
        agent = Agent.from_dict(data)
        assert agent.state_schema == {"messages": {"type": list[ChatMessage], "handler": merge_lists}}

    def test_serde(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
            exit_conditions=["text", "weather_tool"],
            state_schema={"foo": {"type": str}},
        )

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]

        assert serialized_agent["type"] == "haystack.components.agents.agent.Agent"
        assert (
            init_parameters["chat_generator"]["type"]
            == "haystack.components.generators.chat.openai.OpenAIChatGenerator"
        )
        assert init_parameters["streaming_callback"] is None
        assert init_parameters["tools"][0]["data"]["function"] == serialize_callable(weather_function)
        assert (
            init_parameters["tools"][1]["data"]["component"]["type"]
            == "haystack.components.builders.prompt_builder.PromptBuilder"
        )
        assert init_parameters["exit_conditions"] == ["text", "weather_tool"]

        deserialized_agent = Agent.from_dict(serialized_agent)

        assert isinstance(deserialized_agent, Agent)
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert deserialized_agent.tools[0].function is weather_function
        assert isinstance(deserialized_agent.tools[1]._component, PromptBuilder)
        assert deserialized_agent.exit_conditions == ["text", "weather_tool"]
        assert deserialized_agent.state_schema == {
            "foo": {"type": str},
            "messages": {"handler": merge_lists, "type": list[ChatMessage]},
        }

    def test_serde_with_streaming_callback(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(
            chat_generator=generator, tools=[weather_tool, component_tool], streaming_callback=sync_streaming_callback
        )

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]
        assert init_parameters["streaming_callback"] == "test_agent.sync_streaming_callback"

        deserialized_agent = Agent.from_dict(serialized_agent)
        assert deserialized_agent.streaming_callback is sync_streaming_callback

    def test_exit_conditions_validation(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))

        # Test invalid exit condition
        with pytest.raises(ValueError, match="Invalid exit conditions provided:"):
            Agent(chat_generator=generator, tools=[weather_tool, component_tool], exit_conditions=["invalid_tool"])

        # Test default exit condition
        agent = Agent(chat_generator=generator, tools=[weather_tool, component_tool])
        assert agent.exit_conditions == ["text"]

        # Test multiple valid exit conditions
        agent = Agent(
            chat_generator=generator, tools=[weather_tool, component_tool], exit_conditions=["text", "weather_tool"]
        )
        assert agent.exit_conditions == ["text", "weather_tool"]

    def test_tool_concurrency_limit_validation(self, weather_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))

        with pytest.raises(ValueError, match="tool_concurrency_limit must be greater than or equal to 1"):
            Agent(chat_generator=generator, tools=[weather_tool], tool_concurrency_limit=0)

    def test_run_with_params_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        agent = Agent(chat_generator=chat_generator, streaming_callback=streaming_callback, tools=[weather_tool])
        response = agent.run([ChatMessage.from_user("Hello")])

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)

    def test_run_with_run_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))

        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        response = agent.run([ChatMessage.from_user("Hello")], streaming_callback=streaming_callback)

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)

    def test_keep_generator_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        chat_generator = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )

        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        response = agent.run([ChatMessage.from_user("Hello")])

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)

    def test_chat_generator_must_support_tools(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutTools()

        with pytest.raises(TypeError, match="MockChatGeneratorWithoutTools does not accept tools"):
            Agent(chat_generator=chat_generator, tools=[weather_tool])

    def test_no_tools_with_chat_generator_without_tools_support(self):
        chat_generator = MockChatGeneratorWithoutTools()
        agent = Agent(chat_generator=chat_generator, max_agent_steps=1)

        response = agent.run(messages=[ChatMessage.from_user("Hello")])

        assert isinstance(response, dict)
        assert "messages" in response
        assert len(response["messages"]) == 2
        assert response["messages"][0].text == "Hello"
        assert response["messages"][1].text == "Hello"
        assert response["last_message"] == response["messages"][-1]

    def test_exceed_max_steps(self, monkeypatch, weather_tool, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()

        mock_messages = [
            ChatMessage.from_assistant("First response"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
            ),
        ]

        agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=0)

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        with caplog.at_level(logging.WARNING):
            agent.run([ChatMessage.from_user("Hello")])
            assert "Agent reached maximum agent steps" in caplog.text

    def test_exit_condition_exits(self, monkeypatch, weather_tool):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()

        # Mock messages where the exit condition appears in the second message
        mock_messages = [
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
            )
        ]

        agent = Agent(chat_generator=generator, tools=[weather_tool], exit_conditions=["weather_tool"])

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        result = agent.run([ChatMessage.from_user("Hello")])

        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][-2].tool_call.tool_name == "weather_tool"
        assert (
            result["messages"][-1].tool_call_result.result
            == '{"weather": "mostly sunny", "temperature": 7, "unit": "celsius"}'
        )
        assert "last_message" in result
        assert isinstance(result["last_message"], ChatMessage)
        assert result["messages"][-1] == result["last_message"]

    def test_agent_with_no_tools(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()

        # Mock messages where the exit condition appears in the second message
        mock_messages = [ChatMessage.from_assistant("Berlin")]

        with caplog.at_level("WARNING"):
            agent = Agent(chat_generator=generator, tools=[], max_agent_steps=3)
            assert "No tools provided to the Agent." in caplog.text

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        response = agent.run([ChatMessage.from_user("What is the capital of Germany?")])

        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert response["messages"][0].text == "What is the capital of Germany?"
        assert response["messages"][1].text == "Berlin"
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)
        assert response["messages"][-1] == response["last_message"]

    def test_run_with_system_prompt(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        response = agent.run([ChatMessage.from_user("What is the weather in Berlin?")])
        assert response["messages"][0].text == "This is a system prompt."

    def test_run_with_tools_run_param(self, weather_tool: Tool, component_tool: Tool, monkeypatch):
        @component
        class MockChatGenerator:
            tool_invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(
                self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
            ) -> dict[str, Any]:
                assert tools == [weather_tool]
                tool_message = ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
                )
                message = tool_message if not self.tool_invoked else ChatMessage.from_assistant("Hello")
                self.tool_invoked = True
                return {"replies": [message]}

        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[component_tool],
            system_prompt="This is a system prompt.",
            tool_concurrency_limit=3,
            tool_streaming_callback_passthrough=True,
        )
        with patch("haystack.components.agents.agent._run_tool", wraps=_run_tool) as run_tool_mock:
            agent.run([ChatMessage.from_user("What is the weather in Berlin?")], tools=[weather_tool])
        run_tool_mock.assert_called_once()
        assert run_tool_mock.call_args.kwargs["tools"] == [weather_tool]
        assert run_tool_mock.call_args.kwargs["max_workers"] == 3
        assert run_tool_mock.call_args.kwargs["enable_streaming_callback_passthrough"] is True

    def test_run_with_tools_run_param_for_tool_selection(self, weather_tool: Tool, component_tool: Tool, monkeypatch):
        @component
        class MockChatGenerator:
            tool_invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(
                self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
            ) -> dict[str, Any]:
                assert tools == [weather_tool]
                tool_message = ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
                )
                message = tool_message if not self.tool_invoked else ChatMessage.from_assistant("Hello")
                self.tool_invoked = True
                return {"replies": [message]}

        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        with patch("haystack.components.agents.agent._run_tool", wraps=_run_tool) as run_tool_mock:
            agent.run([ChatMessage.from_user("What is the weather in Berlin?")], tools=[weather_tool.name])
        run_tool_mock.assert_called_once()
        assert run_tool_mock.call_args.kwargs["tools"] == [weather_tool]

    def test_run_not_warmed_up(self, weather_tool):
        """Warmup is run automatically on first run"""
        chat_generator = MockChatGeneratorWithoutRunAsync()
        chat_generator.warm_up = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        agent.run([ChatMessage.from_user("What is the weather in Berlin?")])
        assert agent._is_warmed_up is True
        assert chat_generator.warm_up.call_count == 1

    def test_run_no_messages(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[])
        result = agent.run([])
        assert result["messages"] == []

    def test_run_only_system_prompt(self, caplog):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[], system_prompt="This is a system prompt.")
        _ = agent.run([])
        assert "All messages provided to the Agent component are system messages." in caplog.text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run(self, weather_tool):
        chat_generator = OpenAIChatGenerator(model="gpt-4.1-nano")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], max_agent_steps=3)
        response = agent.run([ChatMessage.from_user("What is the weather in Berlin?")])

        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 4
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        # Loose check of message texts
        assert response["messages"][0].text == "What is the weather in Berlin?"
        assert response["messages"][1].text is None
        assert response["messages"][2].text is None
        assert response["messages"][3].text is not None
        # Loose check of message metadata
        assert response["messages"][0].meta == {}
        assert response["messages"][1].meta.get("model") is not None
        assert response["messages"][2].meta == {}
        assert response["messages"][3].meta.get("model") is not None
        # Loose check of tool calls and results
        assert response["messages"][1].tool_calls[0].tool_name == "weather_tool"
        assert response["messages"][1].tool_calls[0].arguments is not None
        assert response["messages"][2].tool_call_results[0].result is not None
        assert response["messages"][2].tool_call_results[0].origin is not None
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)
        assert response["messages"][-1] == response["last_message"]

    @pytest.mark.asyncio
    async def test_generation_kwargs(self):
        chat_generator = MockChatGenerator()

        agent = Agent(chat_generator=chat_generator)

        chat_generator.run_async = AsyncMock(return_value={"replies": [ChatMessage.from_assistant("Hello")]})

        await agent.run_async([ChatMessage.from_user("Hello")], generation_kwargs={"temperature": 0.0})

        expected_messages = [
            ChatMessage(_role=ChatRole.USER, _content=[TextContent(text="Hello")], _name=None, _meta={})
        ]
        chat_generator.run_async.assert_called_once_with(
            messages=expected_messages, generation_kwargs={"temperature": 0.0}, tools=[]
        )

    @pytest.mark.asyncio
    async def test_run_async_uses_chat_generator_run_async_when_available(self, weather_tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        chat_generator.run_async = AsyncMock(
            return_value={"replies": [ChatMessage.from_assistant("Hello from run_async")]}
        )

        result = await agent.run_async([ChatMessage.from_user("Hello")])

        expected_messages = [
            ChatMessage(_role=ChatRole.USER, _content=[TextContent(text="Hello")], _name=None, _meta={})
        ]
        chat_generator.run_async.assert_called_once_with(messages=expected_messages, tools=[weather_tool])

        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in result["messages"]]
        assert "Hello from run_async" in result["messages"][1].text
        assert "last_message" in result
        assert isinstance(result["last_message"], ChatMessage)
        assert result["messages"][-1] == result["last_message"]

    @pytest.mark.asyncio
    async def test_run_async_falls_back_to_sync_run_for_sync_only_chat_generator(self, weather_tool):
        """`agent.run_async` must accept a chat generator that only implements `run` (no `run_async`).
        The Agent should dispatch the sync call to the default executor rather than raising AttributeError."""
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        assert not getattr(chat_generator, "__haystack_supports_async__", False)

        run_mock = MagicMock(wraps=chat_generator.run)
        chat_generator.run = run_mock

        result = await agent.run_async([ChatMessage.from_user("Hello")])

        run_mock.assert_called_once()
        # MockChatGeneratorWithoutRunAsync.run returns ChatMessage.from_assistant("Hello")
        assert result["messages"][1].text == "Hello"
        assert result["last_message"] == result["messages"][-1]

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_agent_streaming_with_tool_call(self, weather_tool):
        chat_generator = OpenAIChatGenerator(model="gpt-4.1-nano")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        result = agent.run(
            [ChatMessage.from_user("What's the weather in Paris?")], streaming_callback=streaming_callback
        )

        assert result is not None
        assert result["messages"] is not None
        assert result["last_message"] is not None
        assert streaming_callback_called

    @pytest.mark.asyncio
    async def test_run_async_with_async_streaming_callback(self, weather_tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=async_streaming_callback)

        # This should not raise any exception
        result = await agent.run_async([ChatMessage.from_user("Hello")])

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][1].text == "Hello from run_async"

    def test_run_with_async_streaming_callback_fails(self, weather_tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=async_streaming_callback)

        with pytest.raises(ValueError, match="The init callback cannot be a coroutine"):
            agent.run([ChatMessage.from_user("Hello")])

    @pytest.mark.asyncio
    async def test_run_async_with_sync_streaming_callback_fails(self, weather_tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=sync_streaming_callback)

        with pytest.raises(ValueError, match="The init callback must be async compatible"):
            await agent.run_async([ChatMessage.from_user("Hello")])


class TestAgentTracing:
    def test_agent_tracing_span_run(self, caplog, monkeypatch, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        _ = agent.run([ChatMessage.from_user("What's the weather in Paris?")])

        # LoggingTracer emits one "Operation: <name>" record when a span exits, immediately followed by that
        # span's tag records. We walk the log stream and bucket tags under the operation that owns them so we
        # can assert the agent's nested span hierarchy, not just a flat list of tags.
        spans: list[tuple[str, dict[str, Any]]] = []
        for record in caplog.records:
            if hasattr(record, "operation_name"):
                spans.append((record.operation_name, {}))
            elif hasattr(record, "tag_name") and spans:
                spans[-1][1][record.tag_name] = record.tag_value

        # Keep only the agent's own spans. With the MockChatGenerator returning no tool calls, the inner
        # `haystack.agent.step.tool` span never fires - the loop exits after the LLM call.
        agent_spans = [(op, tags) for op, tags in spans if op.startswith("haystack.agent")]

        # Exit order (innermost first): LLM child -> step wrapper -> agent.run.
        assert [op for op, _ in agent_spans] == ["haystack.agent.step.llm", "haystack.agent.step", "haystack.agent.run"]

        # LLM child span carries the chat_generator's input/output, nothing else.
        _, llm_tags = agent_spans[0]
        assert set(llm_tags) == {"haystack.agent.step.llm.input", "haystack.agent.step.llm.output"}
        assert (
            llm_tags["haystack.agent.step.llm.input"]
            == '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}'  # noqa: E501
        )
        assert (
            llm_tags["haystack.agent.step.llm.output"]
            == '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}'
        )

        # The step wrapper only carries the iteration counter.
        _, step_tags = agent_spans[1]
        assert step_tags == {"haystack.agent.step": 0}

        # agent.run carries the static config + the final input/output/steps-taken summary.
        _, run_tags = agent_spans[2]
        assert run_tags == {
            "haystack.agent.max_steps": 100,
            "haystack.agent.tools": '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',  # noqa: E501
            "haystack.agent.exit_conditions": '["text"]',
            "haystack.agent.state_schema": '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',  # noqa: E501
            "haystack.agent.input": '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null}',  # noqa: E501
            "haystack.agent.output": '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}',  # noqa: E501
            "haystack.agent.steps_taken": 1,
        }

        # Clean up
        tracing.tracer.is_content_tracing_enabled = False
        tracing.disable_tracing()

    def test_agent_tracing_span_run_with_tool_call(self, caplog, monkeypatch, weather_tool):
        @component
        class ToolCallingChatGenerator:
            tool_invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(
                self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs
            ) -> dict[str, Any]:
                if self.tool_invoked:
                    return {"replies": [ChatMessage.from_assistant("done")]}
                self.tool_invoked = True
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
                        )
                    ]
                }

        agent = Agent(chat_generator=ToolCallingChatGenerator(), tools=[weather_tool])

        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        _ = agent.run([ChatMessage.from_user("What's the weather in Berlin?")])

        spans: list[tuple[str, dict[str, Any]]] = []
        for record in caplog.records:
            if hasattr(record, "operation_name"):
                spans.append((record.operation_name, {}))
            elif hasattr(record, "tag_name") and spans:
                spans[-1][1][record.tag_name] = record.tag_value

        agent_spans = [(op, tags) for op, tags in spans if op.startswith("haystack.agent")]
        assert [op for op, _ in agent_spans] == [
            "haystack.agent.step.llm",
            "haystack.agent.step.tool",
            "haystack.agent.step",
            "haystack.agent.step.llm",
            "haystack.agent.step",
            "haystack.agent.run",
        ]

        _, tool_tags = agent_spans[1]
        assert set(tool_tags) == {"haystack.agent.step.tool.input", "haystack.agent.step.tool.output"}

        _, run_tags = agent_spans[-1]
        assert run_tags["haystack.agent.steps_taken"] == 2

        tracing.tracer.is_content_tracing_enabled = False
        tracing.disable_tracing()

    @pytest.mark.asyncio
    async def test_agent_tracing_span_async_run(self, caplog, monkeypatch, weather_tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        _ = await agent.run_async([ChatMessage.from_user("What's the weather in Paris?")])

        # Bucket each tag under the operation it was emitted from (see sync version for details).
        spans: list[tuple[str, dict[str, Any]]] = []
        for record in caplog.records:
            if hasattr(record, "operation_name"):
                spans.append((record.operation_name, {}))
            elif hasattr(record, "tag_name") and spans:
                spans[-1][1][record.tag_name] = record.tag_value

        agent_spans = [(op, tags) for op, tags in spans if op.startswith("haystack.agent")]

        assert [op for op, _ in agent_spans] == ["haystack.agent.step.llm", "haystack.agent.step", "haystack.agent.run"]

        _, llm_tags = agent_spans[0]
        assert set(llm_tags) == {"haystack.agent.step.llm.input", "haystack.agent.step.llm.output"}
        assert (
            llm_tags["haystack.agent.step.llm.input"]
            == '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}'  # noqa: E501
        )
        assert (
            llm_tags["haystack.agent.step.llm.output"]
            == '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}'  # noqa: E501
        )

        _, step_tags = agent_spans[1]
        assert step_tags == {"haystack.agent.step": 0}

        _, run_tags = agent_spans[2]
        assert run_tags == {
            "haystack.agent.max_steps": 100,
            "haystack.agent.tools": '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',  # noqa: E501
            "haystack.agent.exit_conditions": '["text"]',
            "haystack.agent.state_schema": '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',  # noqa: E501
            "haystack.agent.input": '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null}',  # noqa: E501
            "haystack.agent.output": '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}',  # noqa: E501
            "haystack.agent.steps_taken": 1,
        }

        # Clean up
        tracing.tracer.is_content_tracing_enabled = False
        tracing.disable_tracing()

    def test_agent_tracing_in_pipeline(self, caplog, monkeypatch, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("Hello {{location}}")])
        )
        pipeline.add_component("agent", agent)
        pipeline.connect("prompt_builder.prompt", "agent.messages")

        pipeline.run(data={"prompt_builder": {"location": "Berlin"}})

        # Bucket each tag under the operation it was emitted from (see sync standalone test for details).
        spans: list[tuple[str, dict[str, Any]]] = []
        for record in caplog.records:
            if hasattr(record, "operation_name"):
                spans.append((record.operation_name, {}))
            elif hasattr(record, "tag_name") and spans:
                spans[-1][1][record.tag_name] = record.tag_value

        # Full span hierarchy in exit order (innermost first):
        #   prompt_builder.component.run -> agent.step.llm -> agent.step -> agent.run -> agent.component.run
        #   -> pipeline.run
        assert [op for op, _ in spans] == [
            "haystack.component.run",
            "haystack.agent.step.llm",
            "haystack.agent.step",
            "haystack.agent.run",
            "haystack.component.run",
            "haystack.pipeline.run",
        ]

        # The two `haystack.component.run` spans wrap prompt_builder and agent respectively.
        prompt_builder_component, _ = spans[0], spans[0][1]
        assert prompt_builder_component[1]["haystack.component.name"] == "prompt_builder"
        agent_component = spans[4]
        assert agent_component[1]["haystack.component.name"] == "agent"

        # Agent's own spans: shape and ownership identical to the standalone test.
        agent_spans = [(op, tags) for op, tags in spans if op.startswith("haystack.agent")]
        assert [op for op, _ in agent_spans] == ["haystack.agent.step.llm", "haystack.agent.step", "haystack.agent.run"]

        _, llm_tags = agent_spans[0]
        assert set(llm_tags) == {"haystack.agent.step.llm.input", "haystack.agent.step.llm.output"}

        _, step_tags = agent_spans[1]
        assert step_tags == {"haystack.agent.step": 0}

        _, run_tags = agent_spans[2]
        assert set(run_tags) == {
            "haystack.agent.max_steps",
            "haystack.agent.tools",
            "haystack.agent.exit_conditions",
            "haystack.agent.state_schema",
            "haystack.agent.input",
            "haystack.agent.output",
            "haystack.agent.steps_taken",
        }
        assert run_tags["haystack.agent.steps_taken"] == 1

        # And pipeline.run wraps everything, carrying the pipeline-level summary tags.
        _, pipeline_tags = spans[-1]
        assert set(pipeline_tags) == {
            "haystack.pipeline.input_data",
            "haystack.pipeline.output_data",
            "haystack.pipeline.metadata",
            "haystack.pipeline.max_runs_per_component",
        }

        # Clean up
        tracing.tracer.is_content_tracing_enabled = False
        tracing.disable_tracing()

    def test_agent_span_has_parent_when_in_pipeline(self, spying_tracer, weather_tool):
        """Test that the agent's span has the component span as its parent when running in a pipeline."""
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("Hello {{location}}")])
        )
        pipeline.add_component("agent", agent)
        pipeline.connect("prompt_builder.prompt", "agent.messages")

        pipeline.run(data={"prompt_builder": {"location": "Berlin"}})

        # Find the agent span (haystack.agent.run)
        agent_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.agent.run"]
        assert len(agent_spans) == 1
        agent_span = agent_spans[0]

        # Find the agent's component span (the outer span for the Agent component)
        agent_component_spans = [
            s
            for s in spying_tracer.spans
            if s.operation_name == "haystack.component.run" and s.tags.get("haystack.component.name") == "agent"
        ]
        assert len(agent_component_spans) == 1
        agent_component_span = agent_component_spans[0]

        # Verify the agent span has the component span as its parent
        assert agent_span.parent_span is not None
        assert agent_span.parent_span == agent_component_span


class TestAgentToolSelection:
    def test_tool_selection_by_name(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        result = agent._select_tools([weather_tool.name])
        assert result == [weather_tool]

    def test_tool_selection_new_tool(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        result = agent._select_tools([component_tool])
        assert result == [component_tool]

    def test_tool_selection_existing_tools(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        result = agent._select_tools(None)
        assert result == [weather_tool, component_tool]

    def test_tool_selection_invalid_tool_name(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        with pytest.raises(
            ValueError, match=("The following tool names are not valid: {'invalid_tool_name'}. Valid tool names are: .")
        ):
            agent._select_tools(["invalid_tool_name"])

    def test_tool_selection_no_tools_configured(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[], system_prompt="This is a system prompt.")
        with pytest.raises(ValueError, match="No tools were configured for the Agent at initialization."):
            agent._select_tools([weather_tool.name])

    def test_tool_selection_invalid_type(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        with pytest.raises(
            TypeError,
            match=(
                re.escape(
                    "tools must be a list of Tool and/or Toolset objects, a Toolset, or a list of tool names (strings)."
                )
            ),
        ):
            agent._select_tools("invalid_tool_name")

    def test_tool_selection_with_list_of_toolsets(self, weather_tool: Tool, component_tool: Tool):
        """Test that list of Toolsets and Tools can be passed to agent."""
        chat_generator = MockChatGenerator()
        toolset1 = Toolset([weather_tool])
        standalone_tool = Tool(
            name="standalone",
            description="A standalone tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            function=lambda x: f"Result: {x}",
        )
        toolset2 = Toolset([component_tool])

        agent = Agent(chat_generator=chat_generator, tools=[toolset1, standalone_tool, toolset2])
        result = agent._select_tools(None)

        assert result == [toolset1, standalone_tool, toolset2]
        assert isinstance(result, list)
        assert len(result) == 3

    def test_agent_serde_with_list_of_toolsets(self, weather_tool: Tool, component_tool: Tool, monkeypatch):
        """Test Agent serialization and deserialization with a list of Toolsets."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

        toolset1 = Toolset([weather_tool])
        toolset2 = Toolset([component_tool])

        generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=generator, tools=[toolset1, toolset2])

        serialized_agent = agent.to_dict()

        # Verify serialization preserves list[Toolset] structure
        tools_data = serialized_agent["init_parameters"]["tools"]
        assert isinstance(tools_data, list)
        assert len(tools_data) == 2
        assert all(isinstance(ts, dict) for ts in tools_data)
        assert tools_data[0]["type"] == "haystack.tools.toolset.Toolset"
        assert tools_data[1]["type"] == "haystack.tools.toolset.Toolset"

        # Deserialize and verify
        deserialized_agent = Agent.from_dict(serialized_agent)
        assert isinstance(deserialized_agent.tools, list)
        assert len(deserialized_agent.tools) == 2
        assert all(isinstance(ts, Toolset) for ts in deserialized_agent.tools)


class TestRegisterPromptVariables:
    def test_register_prompt_variables_warning_when_no_prompt_and_required_variables(self, make_agent, caplog):
        make_agent(required_variables=["name"])
        assert "The parameter required_variables is provided but neither" in caplog.text

    def test_register_prompt_variables_set_all_variables_as_required(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("Question: {{question}}"), required_variables="*")
        assert agent._user_chat_prompt_builder.required_variables == "*"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "question" in input_names

    def test_register_prompt_variables_set_required_variables_on_builder(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("Question: {{question}}"), required_variables=["question"])
        assert agent._user_chat_prompt_builder.required_variables == ["question"]

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "question" in input_names

    def test_register_prompt_variables_raises_on_state_schema_conflict(self, make_agent):
        with pytest.raises(
            ValueError, match="Variable 'question' from user_prompt is already defined in the state schema."
        ):
            make_agent(user_prompt=_user_msg("Question: {{question}}"), state_schema={"question": {"type": str}})

    def test_register_prompt_variables_raises_on_run_param_conflict(self, make_agent):
        with pytest.raises(
            ValueError,
            match="Variable 'streaming_callback' from user_prompt conflicts with input names in the run method.",
        ):
            make_agent(user_prompt=_user_msg("{{streaming_callback}} is already a run parameter."))


class TestPrompts:
    def test_system_prompt_incorrect_jinja2_syntax_raises(self, make_agent):
        with pytest.raises(TemplateSyntaxError):
            make_agent(system_prompt="{% message role='system' %}Incomplete syntax.")

    def test_system_prompt_plain_string(self, make_agent):
        agent = make_agent(system_prompt="You are a helpful assistant.")
        assert agent._system_chat_prompt_builder is not None
        result = agent.run(messages=[ChatMessage.from_user("Hi")])
        assert result["messages"][0].is_from(ChatRole.SYSTEM)
        assert result["messages"][0].text == "You are a helpful assistant."

    def test_system_prompt_plain_string_with_template_variables(self, make_agent):
        agent = make_agent(system_prompt="You are an assistant for {{company}}. Your role is {{role}}.")
        assert agent._system_chat_prompt_builder is not None
        assert set(agent._system_chat_prompt_builder.variables) == {"company", "role"}

        result = agent.run(messages=[ChatMessage.from_user("Hi")], company="Acme", role="support agent")
        sys_msg = result["messages"][0]
        assert sys_msg.is_from(ChatRole.SYSTEM)
        assert sys_msg.text == "You are an assistant for Acme. Your role is support agent."

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "company" in input_names
        assert "role" in input_names

    def test_system_prompt_with_template_variables(self, make_agent):
        agent = make_agent(system_prompt=_sys_msg("You are an assistant for {{company}}. Your role is {{role}}."))
        assert agent._system_chat_prompt_builder is not None
        assert set(agent._system_chat_prompt_builder.variables) == {"company", "role"}

        result = agent.run(messages=[ChatMessage.from_user("Hi")], company="Acme", role="support agent")
        sys_msg = result["messages"][0]
        assert sys_msg.is_from(ChatRole.SYSTEM)
        assert sys_msg.text == "You are an assistant for Acme. Your role is support agent."

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "company" in input_names
        assert "role" in input_names

    def test_system_prompt_with_meta(self, make_agent):
        agent = make_agent(
            system_prompt="{% message role='system' meta={'key': 'value'} %}System message with meta{% endmessage %}"
        )
        assert agent._system_chat_prompt_builder is not None

        result = agent.run(messages=[ChatMessage.from_user("Hi")])
        messages = result["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "System message with meta"
        assert messages[0].meta == {"key": "value"}

    def test_user_prompt_only_variables_forwarded_to_builder(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("Question: {{question}}"))
        # 'irrelevant_kwarg' is not a template variable — must not raise
        result = agent.run(messages=[], question="Will it snow?", irrelevant_kwarg="unused")
        assert "messages" in result

    def test_user_prompt_plain_string_with_template_variables(self, make_agent):
        agent = make_agent(user_prompt="Question: {{question}}")
        result = agent.run(messages=[], question="Will it snow?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Question: Will it snow?"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "question" in input_names

    def test_user_prompt_with_template_variables(self, make_agent):
        agent = make_agent(
            user_prompt=_user_msg(
                "Hello {{name|upper}}, check weather for: "
                + "{% for c in cities %}{{c}}{% if not loop.last %}, {% endif %}{% endfor %}"
                + " on {{date}}?"
            )
        )
        result = agent.run(messages=[], name="Alice", cities=["Berlin", "Paris", "Rome"], date="2024-01-15")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Hello ALICE, check weather for: Berlin, Paris, Rome on 2024-01-15?"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "name" in input_names
        assert "cities" in input_names
        assert "date" in input_names

    def test_user_prompt_appended_after_initial_messages(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("And now: {{query}}"))
        initial_messages = [ChatMessage.from_user("First message")]
        result = agent.run(messages=initial_messages, query="What is the weather?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "First message"
        assert user_messages[1].text == "And now: What is the weather?"

    def test_system_prompt_and_user_prompt(self, make_agent):
        agent = make_agent(
            system_prompt=_sys_msg("You help users of {{project}}."),
            user_prompt=_user_msg("Tell me about {{topic}} in the {{project}} context."),
        )
        assert agent._system_chat_prompt_builder is not None
        assert agent._user_chat_prompt_builder is not None

        result = agent.run(messages=[], project="Haystack", topic="pipelines")
        messages = result["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You help users of Haystack."
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Tell me about pipelines in the Haystack context."

    def test_prompt_wrong_role_raises_at_init(self, make_agent):
        with pytest.raises(ValueError, match="system_prompt message block must have role 'system'"):
            make_agent(system_prompt=_user_msg("This is a user message, not system."))

        with pytest.raises(ValueError, match="user_prompt message block must have role 'user'"):
            make_agent(user_prompt=_sys_msg("This is a system message, not user."))

    def test_dynamic_prompt_role_raises_at_runtime(self, make_agent):
        agent = make_agent(user_prompt="{% message role=role_name %}Question: {{question}}{% endmessage %}")
        with pytest.raises(ValueError, match="user_prompt must render to a user message"):
            agent.run(messages=[], role_name="assistant", question="Will it snow?")

    def test_prompt_multiple_message_blocks_raises_at_init(self, make_agent):
        multi_message_prompt = """{% message role='system' %}You are a helpful assistant.{% endmessage %}
        {% message role='user' %}How are you?{% endmessage %}"""

        with pytest.raises(ValueError, match="system_prompt must define exactly one message block"):
            make_agent(system_prompt=multi_message_prompt)

        with pytest.raises(ValueError, match="user_prompt must define exactly one message block"):
            make_agent(user_prompt=multi_message_prompt)


@pytest.mark.integration
class TestAgentUserPromptInPipeline:
    @pytest.fixture
    def document_store_with_docs(self):
        store = InMemoryDocumentStore()
        store.write_documents(
            [
                Document(content="The Eiffel Tower is located in Paris."),
                Document(content="The Brandenburg Gate is in Berlin."),
                Document(content="The Colosseum is in Rome."),
            ]
        )
        return store

    @pytest.fixture
    def make_rag_pipeline(self, document_store_with_docs: InMemoryDocumentStore, make_agent):

        def _factory(user_prompt: str | None = None):
            agent = make_agent(
                user_prompt=user_prompt
                or _user_msg(
                    "Use the following documents to answer the question.\n"
                    "Documents:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}"
                    "Question: {{query}}"
                ),
                system_prompt="You are a knowledgeable assistant.",
                required_variables=["query", "documents"],
            )

            pp = Pipeline()
            pp.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
            pp.add_component("agent", agent)
            pp.connect("retriever.documents", "agent.documents")

            return pp

        return _factory

    def test_rag_pipeline_user_prompt_init_only(self, make_rag_pipeline):
        pipeline = make_rag_pipeline()
        query = "Where is the Colosseum?"
        result = pipeline.run(data={"retriever": {"query": query}, "agent": {"query": query, "messages": []}})
        assert "agent" in result
        agent_output = result["agent"]
        assert "messages" in agent_output
        assert "last_message" in agent_output

        messages = agent_output["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You are a knowledgeable assistant."

        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 1
        rendered = user_messages[0].text
        assert "Question: Where is the Colosseum?" in rendered
        assert "Documents:" in rendered

    def test_rag_pipeline_messages_plus_user_prompt(self, document_store_with_docs, weather_tool):
        chat_generator = MockChatGenerator()

        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool],
            user_prompt=_user_msg("Relevant docs:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}"),
        )
        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("Berlin")]})

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("History: {{history_note}}")])
        )
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
        pipeline.add_component("agent", agent)

        pipeline.connect("prompt_builder.prompt", "agent.messages")
        pipeline.connect("retriever.documents", "agent.documents")

        result = pipeline.run(
            data={
                "prompt_builder": {"history_note": "User previously asked about European cities."},
                "retriever": {"query": "Brandenburg Gate"},
            }
        )
        messages = result["agent"]["messages"]
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert "History:" in user_messages[0].text
        rendered = user_messages[1].text
        assert "Relevant docs:" in rendered


class TestAgentWaitsForBlockedPredecessor:
    """
    Regression test for the scheduling bug introduced by making the 'messages'
    run parameter non-required in https://github.com/deepset-ai/haystack/pull/10638.

    Pipeline shape
    --------------
    Two paths feed into a lazy-variadic joiner that collects messages for the Agent:

        Path A (works):   query → history_parser → messages_joiner.values
        Path B (blocked): files=[] → files_processor (returns {}) → attachments_builder ──╳──→ messages_joiner.values

        messages_joiner.values → agent.messages
        filters → agent.retrieval_filters   (static input from pipeline.run data)

    The bug
    -------
    1. history_parser runs → sends messages to messages_joiner.
    2. files_processor runs with files=[] → returns {} (no output).
    3. attachments_builder is BLOCKED — its mandatory processed_files input never arrives.
    4. messages_joiner gets DEFER: it has a lazy-variadic socket and attachments_builder hasn't executed yet,
       so the joiner doesn't know if more data might still come. It keeps waiting.
    5. agent also gets DEFER: retrieval_filters arrives with sender=None (static pipeline input), which
       satisfies has_any_trigger() on the first visit. The Agent has no mandatory sockets, so can_component_run()
       returns True.
    6. The scheduler tie-breaks DEFER components by topological order, so the joiner should run before the Agent.
       Before the fix the Agent was picked first and executed without messages, raising:

        ValueError("No messages provided to the Agent and neither user_prompt nor system_prompt is set.")
    """

    def test_agent_waits_for_messages_when_predecessor_is_blocked(self, weather_tool):

        @component
        class HistoryParser:
            @component.output_types(messages=list[ChatMessage])
            def run(self, query: str) -> dict:
                return {"messages": [ChatMessage.from_user(query)]}

        @component
        class FilesProcessor:
            """Produces no output when given an empty file list."""

            @component.output_types(processed_files=list[str])
            def run(self, files: list[str]) -> dict:
                if not files:
                    return {}  # _NO_OUTPUT_PRODUCED → blocks AttachmentsBuilder
                return {"processed_files": files}

        @component
        class AttachmentsBuilder:
            """Builds attachment messages; mandatory processed_files from FilesProcessor."""

            @component.output_types(prompt=list[ChatMessage])
            def run(self, processed_files: list[str]) -> dict:
                return {"prompt": [ChatMessage.from_user(f"Files: {processed_files}")]}

        chat_generator = MockChatGenerator()
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool],
            state_schema={"retrieval_filters": {"type": dict[str, Any]}},
        )
        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("done")]})

        pipeline = Pipeline()
        pipeline.add_component("history_parser", HistoryParser())
        pipeline.add_component("files_processor", FilesProcessor())
        pipeline.add_component("attachments_builder", AttachmentsBuilder())
        pipeline.add_component("messages_joiner", ListJoiner(list[ChatMessage]))
        pipeline.add_component("agent", agent)

        pipeline.connect("history_parser.messages", "messages_joiner.values")
        pipeline.connect("files_processor.processed_files", "attachments_builder.processed_files")
        pipeline.connect("attachments_builder.prompt", "messages_joiner.values")
        pipeline.connect("messages_joiner.values", "agent.messages")

        # files=[] → files_processor produces no output → attachments_builder BLOCKED
        # → messages_joiner stays DEFER waiting for the blocked branch
        # → agent (DEFER) must wait for the joiner via topological tie-break
        result = pipeline.run(
            data={
                "history_parser": {"query": "What case law applies?"},
                "files_processor": {"files": []},  # empty → no output
                "agent": {"retrieval_filters": {"field": "date", "value": "2024-01-01"}},
            }
        )
        assert "agent" in result


class TestAgentWarmUp:
    """Tests that Agent.warm_up() correctly warms up tools and toolsets."""

    def _make_tracking_tool(self, name: str = "test_tool") -> Tool:
        tool = Tool(
            name=name,
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "result",
        )
        tool.was_warmed_up = False
        original_warm_up = tool.warm_up

        def tracking_warm_up():
            original_warm_up()
            tool.was_warmed_up = True

        tool.warm_up = tracking_warm_up
        return tool

    def _make_tracking_toolset(self, tools: list) -> Toolset:
        toolset = Toolset(tools)
        toolset.was_warmed_up = False
        original_warm_up = toolset.warm_up

        def tracking_warm_up():
            original_warm_up()
            toolset.was_warmed_up = True

        toolset.warm_up = tracking_warm_up
        return toolset

    def test_warm_up_single_tool(self):
        tool = self._make_tracking_tool()
        agent = Agent(chat_generator=MockChatGenerator(), tools=[tool])

        assert not tool.was_warmed_up
        agent.warm_up()
        assert tool.was_warmed_up

    def test_warm_up_multiple_tools(self):
        tool1 = self._make_tracking_tool("tool1")
        tool2 = self._make_tracking_tool("tool2")
        agent = Agent(chat_generator=MockChatGenerator(), tools=[tool1, tool2])

        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up
        agent.warm_up()
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up

    def test_warm_up_toolset(self):
        inner_tool = self._make_tracking_tool()
        toolset = self._make_tracking_toolset([inner_tool])
        agent = Agent(chat_generator=MockChatGenerator(), tools=toolset)

        assert not toolset.was_warmed_up
        agent.warm_up()
        assert toolset.was_warmed_up

    def test_warm_up_mixed_toolsets(self):
        tool1 = self._make_tracking_tool("tool1")
        toolset1 = self._make_tracking_toolset([tool1])
        tool2 = self._make_tracking_tool("tool2")
        toolset2 = self._make_tracking_toolset([tool2])

        agent = Agent(chat_generator=MockChatGenerator(), tools=toolset1 + toolset2)

        assert not toolset1.was_warmed_up
        assert not toolset2.was_warmed_up
        agent.warm_up()
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up

    def test_warm_up_mixed_list_of_tools_and_toolsets(self):
        tool1 = self._make_tracking_tool("standalone_tool1")
        tool2 = self._make_tracking_tool("standalone_tool2")
        tool3 = self._make_tracking_tool("toolset_tool1")
        toolset1 = self._make_tracking_toolset([tool3])
        tool4 = self._make_tracking_tool("toolset_tool2")
        toolset2 = self._make_tracking_toolset([tool4])

        agent = Agent(chat_generator=MockChatGenerator(), tools=[tool1, toolset1, tool2, toolset2])

        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up
        assert not toolset1.was_warmed_up
        assert not toolset2.was_warmed_up
        agent.warm_up()
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up

    def test_warm_up_is_idempotent(self):
        call_count = {"n": 0}
        tool = Tool(
            name="counting_tool",
            description="A tool that counts warm_up calls",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test",
        )
        original = tool.warm_up

        def counting_warm_up():
            original()
            call_count["n"] += 1

        tool.warm_up = counting_warm_up

        agent = Agent(chat_generator=MockChatGenerator(), tools=[tool])
        agent.warm_up()
        agent.warm_up()
        agent.warm_up()

        assert call_count["n"] == 1

    def test_warm_up_refreshes_toolset(self):
        """Agent.warm_up() must warm up lazy toolsets (e.g. MCPToolset) so the actual tools are available at runtime."""
        placeholder_tool = Tool(
            name="mcp_not_connected_placeholder_123",
            description="Placeholder tool before connection",
            parameters={"type": "object", "properties": {}},
            function=lambda: "placeholder",
        )
        actual_tool = Tool(
            name="get_time",
            description="Get the current time in ISO format",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: "2024-12-01T12:00:00Z",
        )

        class MockMCPToolset(Toolset):
            def __init__(self):
                super().__init__([placeholder_tool])
                self._connected = False

            def warm_up(self):
                if not self._connected:
                    self.tools = [actual_tool]
                    self._connected = True

        mcp_toolset = MockMCPToolset()
        agent = Agent(chat_generator=MockChatGenerator(), tools=mcp_toolset)

        assert mcp_toolset.tools == [placeholder_tool]

        agent.warm_up()

        assert mcp_toolset.tools == [actual_tool]

    def test_run_warms_lazy_toolset_before_tool_selection(self):
        """
        Agent.run() must warm up lazy toolsets before passing tools to the ChatGenerator and before executing
        tool calls.
        """
        placeholder_tool = Tool(
            name="mcp_not_connected_placeholder_123",
            description="Placeholder tool before connection",
            parameters={"type": "object", "properties": {}},
            function=lambda: "placeholder",
        )
        actual_tool = Tool(
            name="get_time",
            description="Get the current time in ISO format",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: "2024-12-01T12:00:00Z",
        )

        class MockMCPToolset(Toolset):
            def __init__(self):
                super().__init__([placeholder_tool])
                self._connected = False

            def warm_up(self):
                if not self._connected:
                    self.tools = [actual_tool]
                    self._connected = True

        @component
        class ToolCallingChatGenerator:
            tool_invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(self, messages: list[ChatMessage], tools: Toolset | None = None, **kwargs) -> dict[str, Any]:
                assert tools is not None
                assert [tool.name for tool in tools] == ["get_time"]
                if self.tool_invoked:
                    return {"replies": [ChatMessage.from_assistant("done")]}
                self.tool_invoked = True
                return {
                    "replies": [ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="get_time", arguments={})])]
                }

        mcp_toolset = MockMCPToolset()
        agent = Agent(chat_generator=ToolCallingChatGenerator(), tools=mcp_toolset)

        result = agent.run([ChatMessage.from_user("What time is it?")])

        assert mcp_toolset.tools == [actual_tool]
        assert result["messages"][2].tool_call_result.result == "2024-12-01T12:00:00Z"
        assert result["last_message"].text == "done"


class TestAgentNotTriggeredByInjectedInput:
    """
    Regression test for https://github.com/deepset-ai/haystack/issues/11109.

    ConditionalRouter routes to `planning`, BranchJoiner never runs, so Agent.messages
    gets no input. A `streaming_callback` injected via `pipeline.run` data must not
    by itself trigger the Agent (would happen if `messages` were optional, since any
    `sender=None` entry flips `has_user_input()` to True).
    """

    def test_agent_not_triggered_by_injected_streaming_callback(self, weather_tool):
        @component
        class Planner:
            @component.output_types(messages=list[ChatMessage], last_role=str)
            def run(self) -> dict:
                return {"messages": [ChatMessage.from_assistant("?")], "last_role": "assistant"}

        chat_generator = MockChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("x")]})

        router = ConditionalRouter(
            routes=[
                {
                    "condition": "{{ last_role == 'tool' }}",
                    "output": "{{ messages }}",
                    "output_name": "processing",
                    "output_type": list[ChatMessage],
                },
                {
                    "condition": "{{ True }}",
                    "output": "{{ messages }}",
                    "output_name": "planning",
                    "output_type": list[ChatMessage],
                },
            ],
            unsafe=True,
        )

        pipeline = Pipeline()
        pipeline.add_component("planner", Planner())
        pipeline.add_component("router", router)
        pipeline.add_component("branch_joiner", BranchJoiner(type_=list[ChatMessage]))
        pipeline.add_component("agent", agent)
        pipeline.connect("planner.messages", "router.messages")
        pipeline.connect("planner.last_role", "router.last_role")
        pipeline.connect("router.processing", "branch_joiner.value")
        pipeline.connect("branch_joiner.value", "agent.messages")

        result = pipeline.run(data={"agent": {"streaming_callback": sync_streaming_callback}})

        assert "agent" not in result
        chat_generator.run.assert_not_called()
