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
from openai import Stream
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk

from haystack import Document, Pipeline, component, tracing
from haystack.components.agents import Agent
from haystack.components.agents.state import merge_lists
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
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
            tool_invoker_kwargs={"max_workers": 5, "enable_streaming_callback_passthrough": True},
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
                                    "required_variables": None,
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
                "tool_invoker_kwargs": {"max_workers": 5, "enable_streaming_callback_passthrough": True},
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
                "tool_invoker_kwargs": None,
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
                                    "required_variables": None,
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
                "tool_invoker_kwargs": {"max_workers": 5, "enable_streaming_callback_passthrough": True},
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
        assert agent.tool_invoker_kwargs == {"max_workers": 5, "enable_streaming_callback_passthrough": True}
        assert agent._tool_invoker.max_workers == 5
        assert agent._tool_invoker.enable_streaming_callback_passthrough is True

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
                "tool_invoker_kwargs": None,
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
                                    "required_variables": None,
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
                "tool_invoker_kwargs": {"max_workers": 5, "enable_streaming_callback_passthrough": True},
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
            == "{'weather': 'mostly sunny', 'temperature': 7, 'unit': 'celsius'}"
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

    def test_run_with_system_prompt_run_param(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(
            chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is the init system prompt."
        )
        response = agent.run(
            [ChatMessage.from_user("What is the weather in Berlin?")], system_prompt="This is the run system prompt."
        )
        assert response["messages"][0].text == "This is the run system prompt."

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
        agent = Agent(chat_generator=chat_generator, tools=[component_tool], system_prompt="This is a system prompt.")
        tool_invoker_run_mock = MagicMock(wraps=agent._tool_invoker.run)
        monkeypatch.setattr(agent._tool_invoker, "run", tool_invoker_run_mock)
        agent.run([ChatMessage.from_user("What is the weather in Berlin?")], tools=[weather_tool])
        tool_invoker_run_mock.assert_called_once()
        assert tool_invoker_run_mock.call_args[1]["tools"] == [weather_tool]

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
        tool_invoker_run_mock = MagicMock(wraps=agent._tool_invoker.run)
        monkeypatch.setattr(agent._tool_invoker, "run", tool_invoker_run_mock)
        agent.run([ChatMessage.from_user("What is the weather in Berlin?")], tools=[weather_tool.name])
        tool_invoker_run_mock.assert_called_once()
        assert tool_invoker_run_mock.call_args[1]["tools"] == [weather_tool]

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
    async def test_run_async_falls_back_to_run_when_chat_generator_has_no_run_async(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()

        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("Hello")]})

        result = await agent.run_async([ChatMessage.from_user("Hello")])

        expected_messages = [
            ChatMessage(_role=ChatRole.USER, _content=[TextContent(text="Hello")], _name=None, _meta={})
        ]
        chat_generator.run.assert_called_once_with(messages=expected_messages, tools=[weather_tool])

        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in result["messages"]]
        assert "Hello" in result["messages"][1].text
        assert "last_message" in result
        assert isinstance(result["last_message"], ChatMessage)
        assert result["messages"][-1] == result["last_message"]

    @pytest.mark.asyncio
    async def test_generation_kwargs(self):
        chat_generator = MockChatGeneratorWithoutRunAsync()

        agent = Agent(chat_generator=chat_generator)

        chat_generator.run = MagicMock(return_value={"replies": [ChatMessage.from_assistant("Hello")]})

        await agent.run_async([ChatMessage.from_user("Hello")], generation_kwargs={"temperature": 0.0})

        expected_messages = [
            ChatMessage(_role=ChatRole.USER, _content=[TextContent(text="Hello")], _name=None, _meta={})
        ]
        chat_generator.run.assert_called_once_with(
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

        # Ensure tracing span was emitted
        assert any("Operation: haystack.component.run" in record.message for record in caplog.records)

        # Check specific tags
        tags_records = [r for r in caplog.records if hasattr(r, "tag_name")]

        expected_tag_names = [
            "haystack.component.name",
            "haystack.component.type",
            "haystack.component.fully_qualified_type",
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.agent.max_steps",
            "haystack.agent.tools",
            "haystack.agent.exit_conditions",
            "haystack.agent.state_schema",
            "haystack.agent.input",
            "haystack.agent.output",
            "haystack.agent.steps_taken",
        ]

        expected_tag_values = [
            "chat_generator",
            "MockChatGeneratorWithoutRunAsync",
            "test_agent.MockChatGeneratorWithoutRunAsync",
            '{"messages": "list", "tools": "list"}',
            '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "senders": []}, "tools": {"type": "list[haystack.tools.tool.Tool] | haystack.tools.toolset.Toolset | None", "senders": []}}',  # noqa: E501
            '{"replies": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "receivers": []}}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}',  # noqa: E501
            1,
            '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}',
            100,
            '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',  # noqa: E501
            '["text"]',
            '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',  # noqa: E501
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null, "break_point": null, "snapshot": null}',  # noqa: E501
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}',  # noqa: E501
            1,
        ]
        for idx, record in enumerate(tags_records):
            assert record.tag_name == expected_tag_names[idx]
            assert record.tag_value == expected_tag_values[idx]

        # Clean up
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

        # Ensure tracing span was emitted
        assert any("Operation: haystack.component.run" in record.message for record in caplog.records)

        # Check specific tags
        tags_records = [r for r in caplog.records if hasattr(r, "tag_name")]

        expected_tag_names = [
            "haystack.component.name",
            "haystack.component.type",
            "haystack.component.fully_qualified_type",
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.agent.max_steps",
            "haystack.agent.tools",
            "haystack.agent.exit_conditions",
            "haystack.agent.state_schema",
            "haystack.agent.input",
            "haystack.agent.output",
            "haystack.agent.steps_taken",
        ]

        expected_tag_values = [
            "chat_generator",
            "MockChatGenerator",
            "test_agent.MockChatGenerator",
            '{"messages": "list", "tools": "list"}',
            '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "senders": []}, "tools": {"type": "list[haystack.tools.tool.Tool] | haystack.tools.toolset.Toolset | None", "senders": []}}',  # noqa: E501
            '{"replies": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "receivers": []}}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}',  # noqa: E501
            1,
            '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}',  # noqa: E501
            100,
            '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',  # noqa: E501
            '["text"]',
            '{"messages": {"type": "list[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',  # noqa: E501
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null, "break_point": null, "snapshot": null}',  # noqa: E501
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}',  # noqa: E501
            1,
        ]
        for idx, record in enumerate(tags_records):
            assert record.tag_name == expected_tag_names[idx]
            assert record.tag_value == expected_tag_values[idx]

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

        assert any("Operation: haystack.pipeline.run" in record.message for record in caplog.records)
        tags_records = [r for r in caplog.records if hasattr(r, "tag_name")]
        expected_tag_names = [
            "haystack.component.name",
            "haystack.component.type",
            "haystack.component.fully_qualified_type",
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.component.name",
            "haystack.component.type",
            "haystack.component.fully_qualified_type",
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.agent.max_steps",
            "haystack.agent.tools",
            "haystack.agent.exit_conditions",
            "haystack.agent.state_schema",
            "haystack.agent.input",
            "haystack.agent.output",
            "haystack.agent.steps_taken",
            "haystack.component.name",
            "haystack.component.type",
            "haystack.component.fully_qualified_type",
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.pipeline.input_data",
            "haystack.pipeline.output_data",
            "haystack.pipeline.metadata",
            "haystack.pipeline.max_runs_per_component",
        ]
        for idx, record in enumerate(tags_records):
            assert record.tag_name == expected_tag_names[idx]

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


def _make_agent_with_user_prompt(
    user_prompt: str, *, chat_generator: MockChatGenerator | None = None, **agent_kwargs
) -> Agent:
    return Agent(chat_generator=chat_generator or MockChatGenerator(), user_prompt=user_prompt, **agent_kwargs)


class TestUserPromptInitialization:
    def test_user_prompt_raises_when_no_messages_and_no_prompt(self, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator(), tools=[weather_tool])
        with pytest.raises(
            ValueError, match="No messages provided to the Agent and neither user_prompt nor system_prompt is set"
        ):
            agent.run()

    def test_user_prompt_conflict_with_state_schema_raises(self, weather_tool):
        with pytest.raises(ValueError, match="already defined in the state schema"):
            _make_agent_with_user_prompt(
                _user_msg("Query: {{custom_field}}"), tools=[weather_tool], state_schema={"custom_field": {"type": str}}
            )

    def test_user_prompt_conflict_with_run_param_raises(self, weather_tool):
        with pytest.raises(ValueError, match="conflicts with input names in the run method"):
            _make_agent_with_user_prompt(_user_msg("{{system_prompt}} is the system prompt."), tools=[weather_tool])

    def test_user_prompt_only_variables_forwarded_to_builder(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Question: {{question}}"), tools=[weather_tool])
        # 'irrelevant_kwarg' is not a template variable — must not raise
        result = agent.run(question="Will it snow?", irrelevant_kwarg="unused")
        assert "messages" in result


class TestUserPromptOnly:
    def test_simple_literal_user_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Tell me the weather."), tools=[weather_tool])
        result = agent.run()
        messages = result["messages"]
        # The rendered user_prompt should be the first (and only) non-system message
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 1
        assert user_messages[0].text == "Tell me the weather."

    def test_user_prompt_with_template_variables(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg(
                "Hello {{name|upper}}, check weather for: "
                + "{% for c in cities %}{{c}}{% if not loop.last %}, {% endif %}{% endfor %}"
                + " on {{date}}?"
            ),
            tools=[weather_tool],
        )
        result = agent.run(name="Alice", cities=["Berlin", "Paris", "Rome"], date="2024-01-15")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Hello ALICE, check weather for: Berlin, Paris, Rome on 2024-01-15?"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "name" in input_names
        assert "cities" in input_names
        assert "date" in input_names

    def test_user_prompt_with_system_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg("What is the weather in {{city}}?"),
            tools=[weather_tool],
            system_prompt="You are a helpful weather assistant.",
        )
        result = agent.run(city="Berlin")
        messages = result["messages"]
        assert messages[0].is_from(ChatRole.SYSTEM)
        assert messages[0].text == "You are a helpful weather assistant."
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "What is the weather in Berlin?"

    def test_user_prompt_with_documents_variable(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg(
                "Answer based on these documents:\n"
                "{% for doc in documents %}{{doc.content}}\n{% endfor %}"
                "Question: {{question}}"
            ),
            tools=[weather_tool],
        )
        docs = [Document(content="Doc A"), Document(content="Doc B")]
        result = agent.run(documents=docs, question="What is in the docs?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert "Doc A" in user_messages[0].text
        assert "Doc B" in user_messages[0].text
        assert "What is in the docs?" in user_messages[0].text

    def test_runtime_user_prompt_overrides_init_prompt(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Default prompt for {{city}}."), tools=[weather_tool])
        result = agent.run(user_prompt=_user_msg("Runtime prompt for {{city}}."), city="Berlin")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Runtime prompt for Berlin."


class TestUserPromptWithMessages:
    def test_user_prompt_appended_after_initial_messages(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("And now: {{query}}"), tools=[weather_tool])
        initial_messages = [ChatMessage.from_user("First message")]
        result = agent.run(messages=initial_messages, query="What is the weather?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "First message"
        assert user_messages[1].text == "And now: What is the weather?"

    def test_runtime_user_prompt_appended_after_initial_messages(self, weather_tool):
        agent = _make_agent_with_user_prompt(_user_msg("Init prompt: {{question}}"), tools=[weather_tool])
        initial_messages = [ChatMessage.from_user("Context message")]
        result = agent.run(
            messages=initial_messages, user_prompt=_user_msg("Follow-up: {{question}}"), question="Is it raining?"
        )
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert len(user_messages) == 2
        assert user_messages[0].text == "Context message"
        assert user_messages[1].text == "Follow-up: Is it raining?"

    def test_messages_plus_user_prompt_with_multiple_kwargs(self, weather_tool):
        agent = _make_agent_with_user_prompt(
            _user_msg("Documents:\n{% for d in documents %}{{d.content}}\n{% endfor %}Q: {{question}}"),
            tools=[weather_tool],
            system_prompt="You are very smart.",
        )
        history = [ChatMessage.from_user("Previous question?"), ChatMessage.from_assistant("Previous answer.")]
        docs = [Document(content="Fact A"), Document(content="Fact B")]
        result = agent.run(messages=history, documents=docs, question="Summarise the facts.")
        messages = result["messages"]
        assert len(messages) == 5

        assert messages[0].role.value == ChatRole.SYSTEM
        assert messages[0].text == "You are very smart."

        assert messages[1].role.value == ChatRole.USER
        assert messages[1].text == "Previous question?"

        assert messages[2].role.value == ChatRole.ASSISTANT
        assert messages[2].text == "Previous answer."

        assert messages[3].role.value == ChatRole.USER
        rendered = messages[3].text
        assert "Fact A" in rendered
        assert "Fact B" in rendered
        assert "Summarise the facts." in rendered

        assert messages[4].role.value == ChatRole.ASSISTANT
        assert messages[4].text == "Hello"


def _make_rag_pipeline(
    document_store_with_docs: InMemoryDocumentStore, weather_tool: Tool, *, user_prompt: str | None = None
):
    agent = _make_agent_with_user_prompt(
        user_prompt=user_prompt
        or _user_msg(
            "Use the following documents to answer the question.\n"
            "Documents:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}"
            "Question: {{query}}"
        ),
        tools=[weather_tool],
        system_prompt="You are a knowledgeable assistant.",
        required_variables=["query", "documents"],
    )

    pp = Pipeline()
    pp.add_component("retriever", InMemoryBM25Retriever(document_store=document_store_with_docs))
    pp.add_component("agent", agent)
    pp.connect("retriever.documents", "agent.documents")

    return pp


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

    def test_rag_pipeline_user_prompt_init_only(self, document_store_with_docs, weather_tool):
        pipeline = _make_rag_pipeline(document_store_with_docs, weather_tool)
        query = "Where is the Colosseum?"
        result = pipeline.run(data={"retriever": {"query": query}, "agent": {"query": query}})
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

    def test_rag_pipeline_user_prompt_runtime_override(self, document_store_with_docs, weather_tool):
        user_prompt = _user_msg(
            "Documents:\n{% for doc in documents %}{{doc.content}}\n{% endfor %}Question: {{query}}"
        )
        pipeline = _make_rag_pipeline(document_store_with_docs, weather_tool, user_prompt=user_prompt)

        query = "Where is the Eiffel Tower?"
        result = pipeline.run(
            data={
                "retriever": {"query": query},
                "agent": {
                    "user_prompt": _user_msg(
                        "OVERRIDE: Using docs:\n"
                        "{% for doc in documents %}{{doc.content}}\n{% endfor %}"
                        "Answer: {{query}}"
                    ),
                    "query": query,
                },
            }
        )
        messages = result["agent"]["messages"]
        user_messages = [m for m in messages if m.is_from(ChatRole.USER)]
        rendered = user_messages[0].text
        assert "OVERRIDE:" in rendered
        assert "Where is the Eiffel Tower?" in rendered

    def test_rag_pipeline_messages_plus_user_prompt(self, document_store_with_docs, weather_tool):
        from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder

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
