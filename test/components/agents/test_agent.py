# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime
from typing import Iterator, Dict, Any, List, Optional, Union

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from openai import Stream
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk

from haystack.tracing.logging_tracer import LoggingTracer
from haystack import Pipeline, tracing
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.component.types import OutputSocket
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.chat_message import ChatRole, TextContent
from haystack.dataclasses.streaming_chunk import StreamingChunk

from haystack.tools import Tool, ComponentTool
from haystack.tools.toolset import Toolset
from haystack.utils import serialize_callable, Secret
from haystack.components.agents.state import merge_lists


def streaming_callback_for_serde(chunk: StreamingChunk):
    pass


def weather_function(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


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
        super().__init__(client=client, *args, **kwargs)
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


class MockChatGeneratorWithoutTools(ChatGenerator):
    """A mock chat generator that implements ChatGenerator protocol but doesn't support tools."""

    __haystack_input__ = MagicMock(_sockets_dict={})
    __haystack_output__ = MagicMock(_sockets_dict={})

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutTools", "data": {}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockChatGeneratorWithoutTools":
        return cls()

    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


class MockChatGeneratorWithoutRunAsync(ChatGenerator):
    """A mock chat generator that implements ChatGenerator protocol but doesn't have run_async method."""

    __haystack_input__ = MagicMock(_sockets_dict={})
    __haystack_output__ = MagicMock(_sockets_dict={})

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutRunAsync", "data": {}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockChatGeneratorWithoutRunAsync":
        return cls()

    def run(
        self, messages: List[ChatMessage], tools: Optional[Union[List[Tool], Toolset]] = None, **kwargs
    ) -> Dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


class MockChatGeneratorWithRunAsync(ChatGenerator):
    __haystack_supports_async__ = True
    __haystack_input__ = MagicMock(_sockets_dict={})
    __haystack_output__ = MagicMock(_sockets_dict={})

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "MockChatGeneratorWithoutRunAsync", "data": {}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockChatGeneratorWithoutRunAsync":
        return cls()

    def run(
        self, messages: List[ChatMessage], tools: Optional[Union[List[Tool], Toolset]] = None, **kwargs
    ) -> Dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}

    async def run_async(
        self, messages: List[ChatMessage], tools: Optional[Union[List[Tool], Toolset]] = None, **kwargs
    ) -> Dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello from run_async")]}


class TestAgent:
    def test_output_types(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool, component_tool])
        assert agent.__haystack_output__._sockets_dict == {
            "messages": OutputSocket(name="messages", type=List[ChatMessage], receivers=[]),
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
        )
        serialized_agent = agent.to_dict()
        assert serialized_agent == {
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
                "state_schema": {"foo": {"type": "str"}},
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
            },
        }

    def test_to_dict_with_toolset(self, monkeypatch, weather_tool):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        toolset = Toolset(tools=[weather_tool])
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=toolset)
        serialized_agent = agent.to_dict()
        assert serialized_agent == {
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
            },
        }

    def test_from_dict(self, monkeypatch):
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
                "state_schema": {"foo": {"type": "str"}},
                "max_agent_steps": 100,
                "raise_on_tool_invocation_failure": False,
                "streaming_callback": None,
            },
        }
        agent = Agent.from_dict(data)
        assert isinstance(agent, Agent)
        assert isinstance(agent.chat_generator, OpenAIChatGenerator)
        assert agent.chat_generator.model == "gpt-4o-mini"
        assert agent.chat_generator.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert agent.tools[0].function is weather_function
        assert isinstance(agent.tools[1]._component, PromptBuilder)
        assert agent.exit_conditions == ["text", "weather_tool"]
        assert agent.state_schema == {
            "foo": {"type": str},
            "messages": {"handler": merge_lists, "type": List[ChatMessage]},
        }

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
            },
        }
        agent = Agent.from_dict(data)
        assert isinstance(agent, Agent)
        assert isinstance(agent.chat_generator, OpenAIChatGenerator)
        assert agent.chat_generator.model == "gpt-4o-mini"
        assert agent.chat_generator.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert isinstance(agent.tools, Toolset)
        assert agent.tools[0].function is weather_function
        assert agent.exit_conditions == ["text"]

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
            "messages": {"handler": merge_lists, "type": List[ChatMessage]},
        }

    def test_serde_with_streaming_callback(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
            streaming_callback=streaming_callback_for_serde,
        )

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]
        assert init_parameters["streaming_callback"] == "test_agent.streaming_callback_for_serde"

        deserialized_agent = Agent.from_dict(serialized_agent)
        assert deserialized_agent.streaming_callback is streaming_callback_for_serde

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
        agent.warm_up()
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
        agent.warm_up()
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
        agent.warm_up()
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

    def test_multiple_llm_responses_with_tool_call(self, monkeypatch, weather_tool):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()

        mock_messages = [
            ChatMessage.from_assistant("First response"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
            ),
        ]

        agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=1)
        agent.warm_up()

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        result = agent.run([ChatMessage.from_user("Hello")])

        assert "messages" in result
        assert len(result["messages"]) == 4
        assert (
            result["messages"][-1].tool_call_result.result
            == "{'weather': 'mostly sunny', 'temperature': 7, 'unit': 'celsius'}"
        )
        assert "last_message" in result
        assert isinstance(result["last_message"], ChatMessage)
        assert result["messages"][-1] == result["last_message"]

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
        agent.warm_up()

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        with caplog.at_level(logging.WARNING):
            agent.run([ChatMessage.from_user("Hello")])
            assert "Agent reached maximum agent steps" in caplog.text

    def test_exit_conditions_checked_across_all_llm_messages(self, monkeypatch, weather_tool):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        generator = OpenAIChatGenerator()

        # Mock messages where the exit condition appears in the second message
        mock_messages = [
            ChatMessage.from_assistant("First response"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
            ),
        ]

        agent = Agent(chat_generator=generator, tools=[weather_tool], exit_conditions=["weather_tool"])
        agent.warm_up()

        # Patch agent.chat_generator.run to return mock_messages
        agent.chat_generator.run = MagicMock(return_value={"replies": mock_messages})

        result = agent.run([ChatMessage.from_user("Hello")])

        assert "messages" in result
        assert len(result["messages"]) == 4
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
            agent.warm_up()
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
        agent.warm_up()
        response = agent.run([ChatMessage.from_user("What is the weather in Berlin?")])
        assert response["messages"][0].text == "This is a system prompt."

    def test_run_not_warmed_up(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        chat_generator.warm_up = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        with pytest.raises(RuntimeError, match="The component Agent wasn't warmed up."):
            agent.run([ChatMessage.from_user("What is the weather in Berlin?")])

    def test_run_no_messages(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[])
        agent.warm_up()
        result = agent.run([])
        assert result["messages"] == []

    def test_run_only_system_prompt(self, caplog):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[], system_prompt="This is a system prompt.")
        agent.warm_up()
        _ = agent.run([])
        assert "All messages provided to the Agent component are system messages." in caplog.text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run(self, weather_tool):
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], max_agent_steps=3)
        agent.warm_up()
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
        agent.warm_up()

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
    async def test_run_async_uses_chat_generator_run_async_when_available(self, weather_tool):
        chat_generator = MockChatGeneratorWithRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        agent.warm_up()

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
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        agent.warm_up()
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
            '{"messages": "list", "tools": "list"}',
            "{}",
            "{}",
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}',
            1,
            '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}',
            100,
            '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',
            '["text"]',
            '{"messages": {"type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello"}]}]}',
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
        chat_generator = MockChatGeneratorWithRunAsync()
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
            "MockChatGeneratorWithRunAsync",
            '{"messages": "list", "tools": "list"}',
            "{}",
            "{}",
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "tools": [{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]}',
            1,
            '{"replies": [{"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}',
            100,
            '[{"type": "haystack.tools.tool.Tool", "data": {"name": "weather_tool", "description": "Provides weather information for a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}, "function": "test_agent.weather_function", "outputs_to_string": null, "inputs_from_state": null, "outputs_to_state": null}}]',
            '["text"]',
            '{"messages": {"type": "typing.List[haystack.dataclasses.chat_message.ChatMessage]", "handler": "haystack.components.agents.state.state_utils.merge_lists"}}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}], "streaming_callback": null}',
            '{"messages": [{"role": "user", "meta": {}, "name": null, "content": [{"text": "What\'s the weather in Paris?"}]}, {"role": "assistant", "meta": {}, "name": null, "content": [{"text": "Hello from run_async"}]}]}',
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
        agent.warm_up()

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
            "haystack.component.input_types",
            "haystack.component.input_spec",
            "haystack.component.output_spec",
            "haystack.component.input",
            "haystack.component.visits",
            "haystack.component.output",
            "haystack.component.name",
            "haystack.component.type",
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
