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

from haystack import Document, Pipeline, component
from haystack.components.agents.agent import Agent
from haystack.components.agents.state import State, merge_lists, replace_values
from haystack.components.agents.tool_calling import _run_tool
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat import MockChatGenerator
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
from haystack.hooks import hook
from haystack.tools import ComponentTool, Tool
from haystack.tools.toolset import Toolset
from haystack.utils import Secret


def _user_msg(text: str) -> str:
    return f'{{% message role="user" %}}{text}{{% endmessage %}}'


def _sys_msg(text: str) -> str:
    return f'{{% message role="system" %}}{text}{{% endmessage %}}'


def _assistant_with_usage(text: str | None = None, *, tool_calls=None, usage: dict[str, Any] | None = None):
    """Build an assistant ChatMessage with optional tool_calls and `meta['usage']` populated."""
    meta: dict[str, Any] = {}
    if usage is not None:
        meta["usage"] = usage
    if tool_calls is not None:
        return ChatMessage.from_assistant(tool_calls=tool_calls, meta=meta or None)
    return ChatMessage.from_assistant(text or "", meta=meta or None)


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
        return Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], **kwargs)

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
class ToolAssertingChatGenerator:
    """Asserts the Agent forwards the expected tools, then drives one tool call before a plain reply."""

    def __init__(self, expected_tools):
        self.expected_tools = expected_tools
        self.tool_invoked = False

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        assert tools == self.expected_tools
        tool_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        message = tool_message if not self.tool_invoked else ChatMessage.from_assistant("Hello")
        self.tool_invoked = True
        return {"replies": [message]}


def _parallel_tool_calling_generator() -> MockChatGenerator:
    """Requests two `weather_tool` calls on the first turn, then returns a plain reply so the agent loop exits."""
    return MockChatGenerator(
        [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
                    ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
                ]
            ),
            "done",
        ]
    )


class TestAgentInit:
    def test_state_schema_resolution(self, weather_tool):
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], state_schema={"foo": {"type": str}}
        )

        assert agent.state_schema == {"foo": {"type": str}}
        assert agent.resolved_state_schema == {
            "foo": {"type": str},
            "messages": {"type": list[ChatMessage], "handler": merge_lists},
            "step_count": {"type": int, "handler": replace_values},
            "token_usage": {"type": dict[str, Any], "handler": replace_values},
            "tool_call_counts": {"type": dict[str, int], "handler": replace_values},
            "exit_reason": {"type": str, "handler": replace_values},
            "continue_run": {"type": bool, "handler": replace_values},
            "tools": {"type": list, "handler": replace_values},
            "hook_context": {"type": dict[str, Any], "handler": replace_values},
            "context_tokens": {"type": int, "handler": replace_values},
        }

    def test_output_types(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        chat_generator = OpenAIChatGenerator()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool, component_tool])
        assert agent.__haystack_output__._sockets_dict == {
            "messages": OutputSocket(name="messages", type=list[ChatMessage], receivers=[]),
            "last_message": OutputSocket(name="last_message", type=ChatMessage, receivers=[]),
            "step_count": OutputSocket(name="step_count", type=int, receivers=[]),
            "token_usage": OutputSocket(name="token_usage", type=dict[str, Any], receivers=[]),
            "tool_call_counts": OutputSocket(name="tool_call_counts", type=dict[str, int], receivers=[]),
            "exit_reason": OutputSocket(name="exit_reason", type=str, receivers=[]),
        }
        # Check that the run-metadata keys are not set up as input sockets
        assert {"step_count", "token_usage", "tool_call_counts", "exit_reason"}.isdisjoint(
            agent.__haystack_input__._sockets_dict.keys()
        )
        # Internal-only state keys (those that are not also run parameters) are exposed as neither inputs nor outputs.
        for internal_key in ("continue_run", "context_tokens"):
            assert internal_key not in agent.__haystack_input__._sockets_dict
            assert internal_key not in agent.__haystack_output__._sockets_dict

    def test_reserved_state_schema_keys_raise(self, weather_tool):
        for reserved in ("step_count", "token_usage", "context_tokens", "tool_call_counts", "exit_reason"):
            with pytest.raises(ValueError, match="reserved for Agent internal state"):
                Agent(
                    chat_generator=MockChatGenerator("Hello"),
                    tools=[weather_tool],
                    state_schema={reserved: {"type": int}},
                )

    def test_exit_conditions(self, weather_tool, component_tool):
        # Default exit condition
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool, component_tool])
        assert agent.exit_conditions == ["text"]

        # Multiple exit conditions are stored as-is
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"),
            tools=[weather_tool, component_tool],
            exit_conditions=["text", "weather_tool"],
        )
        assert agent.exit_conditions == ["text", "weather_tool"]

        # Exit conditions are no longer validated against tool names at init: tool sets can be dynamic
        # (e.g. SearchableToolset/MCPToolset) or provided at runtime, so unknown names pass through.
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], exit_conditions=["not_loaded_yet"]
        )
        assert agent.exit_conditions == ["not_loaded_yet"]

    def test_tool_concurrency_limit_validation(self, weather_tool):
        with pytest.raises(ValueError, match="tool_concurrency_limit must be greater than or equal to 1"):
            Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], tool_concurrency_limit=0)

    def test_chat_generator_must_support_tools(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutTools()

        with pytest.raises(TypeError, match="MockChatGeneratorWithoutTools does not accept tools"):
            Agent(chat_generator=chat_generator, tools=[weather_tool])


class TestAgentSerialization:
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
                            "function": "agents.test_agent.weather_function",
                            "async_function": None,
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
                "required_variables": "*",
                "exit_conditions": ["text", "weather_tool"],
                "state_schema": {"foo": {"type": "str"}},
                "max_agent_steps": 100,
                "streaming_callback": None,
                "raise_on_tool_invocation_failure": False,
                "tool_concurrency_limit": 5,
                "tool_streaming_callback_passthrough": True,
                "hooks": None,
            },
        }
        assert serialized_agent == expected_structure

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
                            "function": "agents.test_agent.weather_function",
                            "async_function": None,
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
        assert agent.state_schema == {"foo": {"type": str}}
        assert agent.tool_concurrency_limit == 5
        assert agent.tool_streaming_callback_passthrough is True

    def test_from_dict_state_schema_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        data = {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {"model": "gpt-4o-mini"},
                },
                "state_schema": None,
            },
        }
        agent = Agent.from_dict(data)
        assert agent.state_schema == {}

    def test_serde(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("FAKE_OPENAI_KEY", "fake-key")
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
            exit_conditions=["text", "weather_tool"],
            state_schema={"foo": {"type": str}},
            streaming_callback=sync_streaming_callback,
        )

        deserialized_agent = Agent.from_dict(agent.to_dict())

        assert deserialized_agent.to_dict() == agent.to_dict()
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert deserialized_agent.tools[0].function is weather_function
        assert isinstance(deserialized_agent.tools[1]._component, PromptBuilder)
        assert deserialized_agent.streaming_callback is sync_streaming_callback

    def test_serde_with_toolset(self, weather_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=Toolset(tools=[weather_tool]))

        restored = Agent.from_dict(agent.to_dict())

        assert isinstance(restored.tools, Toolset)
        assert restored.tools[0].function is weather_function

    def test_serde_with_list_of_toolsets(self, weather_tool, component_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[Toolset([weather_tool]), Toolset([component_tool])])

        restored = Agent.from_dict(agent.to_dict())

        assert isinstance(restored.tools, list)
        assert len(restored.tools) == 2
        assert all(isinstance(ts, Toolset) for ts in restored.tools)
        assert restored.tools[0][0].function is weather_function


class TestAgentClone:
    def test_clone(self, weather_tool):
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"),
            tools=[weather_tool],
            system_prompt="You are helpful",
            exit_conditions=["text", "weather_tool"],
            state_schema={"foo": {"type": str}},
            max_agent_steps=7,
        )

        clone = agent.clone()

        assert clone is not agent
        assert clone.to_dict() == agent.to_dict()

    @pytest.mark.parametrize(
        "name, value",
        [
            ("system_prompt", "A nice system prompt"),
            ("max_agent_steps", 3),
            ("exit_conditions", ["weather_tool"]),
            ("state_schema", {"bar": {"type": int}}),
        ],
    )
    def test_clone_with_overrides(self, weather_tool, name, value):
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"),
            tools=[weather_tool],
            system_prompt="You are helpful",
            state_schema={"foo": {"type": str}},
        )

        clone = agent.clone(**{name: value})

        assert getattr(clone, name) == value

        # only the overridden init parameter differs
        original_params = agent.to_dict()["init_parameters"]
        clone_params = clone.to_dict()["init_parameters"]
        assert clone_params.keys() == original_params.keys()
        for key in original_params:
            if key == name:
                assert clone_params[key] != original_params[key]
            else:
                assert clone_params[key] == original_params[key]

    def test_clone_with_additional_state_schema_and_tools(self, weather_tool, component_tool):
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], state_schema={"foo": {"type": str}}
        )

        clone = agent.clone(
            tools=[*agent.tools, component_tool], state_schema={**agent.state_schema, "notes": {"type": str}}
        )

        assert clone.tools == [weather_tool, component_tool]
        assert clone.state_schema == {"foo": {"type": str}, "notes": {"type": str}}


class TestAgentRun:
    def test_agent_with_no_tools(self):
        agent = Agent(chat_generator=MockChatGenerator("Berlin"), tools=[], max_agent_steps=3)

        response = agent.run([ChatMessage.from_user("What is the capital of Germany?")])

        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert response["messages"][0].text == "What is the capital of Germany?"
        assert response["messages"][1].text == "Berlin"
        assert "last_message" in response
        assert isinstance(response["last_message"], ChatMessage)
        assert response["messages"][-1] == response["last_message"]
        # With no tools the loop always exits after the first reply, reporting the "text" exit reason.
        assert response["exit_reason"] == "text"

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

    def test_run_with_system_prompt(self, weather_tool):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        response = agent.run([ChatMessage.from_user("What is the weather in Berlin?")])
        assert response["messages"][0].text == "This is a system prompt."

    def test_run_only_system_prompt(self, caplog):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        agent = Agent(chat_generator=chat_generator, tools=[], system_prompt="This is a system prompt.")
        _ = agent.run([])
        assert "All messages provided to the Agent component are system messages." in caplog.text

    def test_run_no_messages(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[])
        result = agent.run([])
        assert result["messages"] == []

    def test_run_with_tools_run_param(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = ToolAssertingChatGenerator(expected_tools=[weather_tool])
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

    def test_run_with_tools_run_param_for_tool_selection(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = ToolAssertingChatGenerator(expected_tools=[weather_tool])
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        with patch("haystack.components.agents.agent._run_tool", wraps=_run_tool) as run_tool_mock:
            agent.run([ChatMessage.from_user("What is the weather in Berlin?")], tools=[weather_tool.name])
        run_tool_mock.assert_called_once()
        assert run_tool_mock.call_args.kwargs["tools"] == [weather_tool]

    @pytest.mark.asyncio
    async def test_generation_kwargs(self):
        chat_generator = MockChatGenerator("Hello")

        agent = Agent(chat_generator=chat_generator)

        chat_generator.run_async = AsyncMock(return_value={"replies": [ChatMessage.from_assistant("Hello")]})

        await agent.run_async([ChatMessage.from_user("Hello")], generation_kwargs={"temperature": 0.0})

        expected_messages = [
            ChatMessage(_role=ChatRole.USER, _content=[TextContent(text="Hello")], _name=None, _meta={})
        ]
        # No tools were configured, so the Agent does not pass a `tools` argument to the chat generator.
        chat_generator.run_async.assert_called_once_with(
            messages=expected_messages, generation_kwargs={"temperature": 0.0}
        )

    @pytest.mark.asyncio
    async def test_run_async_uses_chat_generator_run_async_when_available(self, weather_tool):
        chat_generator = MockChatGenerator("Hello")
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

    def test_run_populates_token_usage_and_tool_call_counts(self, weather_tool):
        """A multi-step run aggregates step_count, token_usage, and tool_call_counts."""
        first_step = [
            _assistant_with_usage(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
        second_step = [
            _assistant_with_usage("Done.", usage={"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9})
        ]
        agent = Agent(chat_generator=MockChatGenerator(first_step + second_step), tools=[weather_tool])

        result = agent.run([ChatMessage.from_user("Hi")])
        assert result["step_count"] == 2
        assert result["tool_call_counts"] == {"weather_tool": 1}
        assert result["token_usage"] == {"prompt_tokens": 16, "completion_tokens": 8, "total_tokens": 24}

    @pytest.mark.asyncio
    async def test_run_async_populates_token_usage_and_tool_call_counts(self, weather_tool):
        first_step = [
            _assistant_with_usage(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            )
        ]
        second_step = [
            _assistant_with_usage("Done.", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
        ]
        agent = Agent(chat_generator=MockChatGenerator(first_step + second_step), tools=[weather_tool])

        result = await agent.run_async([ChatMessage.from_user("Hi")])
        assert result["step_count"] == 2
        assert result["tool_call_counts"] == {"weather_tool": 1}
        assert result["token_usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    def test_metadata_outputs_show_defaults_when_no_data(self, weather_tool):
        """`token_usage` stays empty and `tool_call_counts` reports zero for every tool when nothing happens."""
        # A text-only reply whose `usage` meta is empty leaves `token_usage` empty after aggregation.
        chat_generator = MockChatGenerator(ChatMessage.from_assistant("Hello", meta={"usage": {}}))
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        result = agent.run([ChatMessage.from_user("Hi")])
        assert result["step_count"] == 1
        assert result["token_usage"] == {}
        assert result["tool_call_counts"] == {"weather_tool": 0}

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
        # Auto-populated run outputs:
        # 4 messages → tool call + final answer = 2 LLM calls = 2 steps; one weather_tool invocation.
        assert response["step_count"] == 2
        assert response["tool_call_counts"] == {"weather_tool": 1}
        assert response["token_usage"]["prompt_tokens"] > 0
        assert response["token_usage"]["completion_tokens"] > 0
        assert response["token_usage"]["total_tokens"] > 0


class TestAgentStreaming:
    def test_run_with_params_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=streaming_callback)
        response = agent.run([ChatMessage.from_user("Hello")])

        assert streaming_callback_called is True
        assert len(response["messages"]) == 2
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert response["last_message"] == response["messages"][-1]

    def test_run_with_run_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        response = agent.run([ChatMessage.from_user("Hello")], streaming_callback=streaming_callback)

        assert streaming_callback_called is True
        assert len(response["messages"]) == 2
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert response["last_message"] == response["messages"][-1]

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

        assert streaming_callback_called is True
        assert len(response["messages"]) == 2
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
        assert response["last_message"] == response["messages"][-1]

    def test_run_with_async_streaming_callback_fails(self, weather_tool):
        chat_generator = MockChatGenerator("Hello")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=async_streaming_callback)

        with pytest.raises(ValueError, match="The init callback cannot be a coroutine"):
            agent.run([ChatMessage.from_user("Hello")])

    @pytest.mark.asyncio
    async def test_run_async_with_async_streaming_callback(self, weather_tool):
        chat_generator = MockChatGenerator("Hello")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=async_streaming_callback)

        # This should not raise any exception
        result = await agent.run_async([ChatMessage.from_user("Hello")])

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][1].text == "Hello"

    @pytest.mark.asyncio
    async def test_run_async_with_sync_streaming_callback_warns(self, weather_tool, caplog):
        chat_generator = MockChatGenerator("Hello")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], streaming_callback=sync_streaming_callback)

        with caplog.at_level(logging.WARNING):
            result = await agent.run_async([ChatMessage.from_user("Hello")])

        assert "sync streaming callback" in caplog.text
        assert "messages" in result
        assert len(result["messages"]) == 2

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
            [ChatMessage.from_user("What's the weather in Paris?")],
            streaming_callback=streaming_callback,
            generation_kwargs={"stream_options": {"include_usage": True}},
        )

        assert result is not None
        assert result["messages"] is not None
        assert result["last_message"] is not None
        assert streaming_callback_called
        # Auto-populated run outputs.
        assert result["step_count"] == 2
        assert result["tool_call_counts"] == {"weather_tool": 1}
        assert result["token_usage"]["prompt_tokens"] > 0
        assert result["token_usage"]["completion_tokens"] > 0
        assert result["token_usage"]["total_tokens"] > 0


class TestAgentContextTokens:
    """The Agent refreshes the internal `context_tokens` after each LLM call so a hook can read the current
    context-window size (e.g. to trigger compaction). It is a per-call snapshot, not accumulated."""

    def test_before_llm_hook_reads_refreshed_context_tokens(self, weather_tool):
        # Step 1: a tool call (prompt 10 + completion 5 = 15). Step 2: the final text answer.
        first_step = [
            _assistant_with_usage(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
        second_step = [_assistant_with_usage("Done.", usage={"prompt_tokens": 20, "completion_tokens": 8})]

        seen: list[int] = []

        @hook
        def capture(state: State) -> None:
            seen.append(state.get("context_tokens"))

        agent = Agent(
            chat_generator=MockChatGenerator(first_step + second_step),
            tools=[weather_tool],
            hooks={"before_llm": [capture]},
        )
        agent.run([ChatMessage.from_user("Weather in Berlin?")])

        # Before the first call there is no usage yet (0); before the second call it reflects the first call (15),
        # confirming the recorder runs in the loop and the value is refreshed per call rather than accumulated.
        assert seen == [0, 15]

    @pytest.mark.asyncio
    async def test_before_llm_hook_reads_refreshed_context_tokens_async(self, weather_tool):
        first_step = [
            _assistant_with_usage(
                tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
        second_step = [_assistant_with_usage("Done.", usage={"prompt_tokens": 20, "completion_tokens": 8})]

        seen: list[int] = []

        @hook
        def capture(state: State) -> None:
            seen.append(state.get("context_tokens"))

        agent = Agent(
            chat_generator=MockChatGenerator(first_step + second_step),
            tools=[weather_tool],
            hooks={"before_llm": [capture]},
        )
        await agent.run_async([ChatMessage.from_user("Weather in Berlin?")])
        assert seen == [0, 15]


class TestAgentExitConditions:
    def test_check_exit_conditions_parallel_tool_calls(self, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], exit_conditions=["weather_tool"])

        finish_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        other_call = ToolCall(tool_name="search", arguments={"q": "weather Berlin"})

        # Exit-condition call first
        llm_first = [ChatMessage.from_assistant(tool_calls=[finish_call, other_call])]
        # Exit-condition call second
        llm_second = [ChatMessage.from_assistant(tool_calls=[other_call, finish_call])]
        tool_messages_ok = [ChatMessage.from_tool(tool_result="ok", origin=finish_call, error=False)]

        assert agent._check_exit_conditions(llm_first, tool_messages_ok) == "weather_tool"
        assert agent._check_exit_conditions(llm_second, tool_messages_ok) == "weather_tool"

    def test_check_exit_conditions_parallel_calls_with_errored_exit_tool(self, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], exit_conditions=["weather_tool"])

        finish_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        other_call = ToolCall(tool_name="search", arguments={"q": "weather Berlin"})

        llm_messages = [ChatMessage.from_assistant(tool_calls=[other_call, finish_call])]
        tool_messages_errored = [ChatMessage.from_tool(tool_result="boom", origin=finish_call, error=True)]

        assert agent._check_exit_conditions(llm_messages, tool_messages_errored) is None

    def test_check_exit_conditions_parallel_calls_error_only_on_non_exit_tool(self, weather_tool, component_tool):
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"),
            tools=[weather_tool, component_tool],
            exit_conditions=["weather_tool"],
        )

        finish_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        other_call = ToolCall(tool_name="parrot", arguments={"parrot": "hi"})

        llm_messages = [ChatMessage.from_assistant(tool_calls=[other_call, finish_call])]
        tool_messages = [
            ChatMessage.from_tool(tool_result="boom", origin=other_call, error=True),
            ChatMessage.from_tool(tool_result="ok", origin=finish_call, error=False),
        ]

        assert agent._check_exit_conditions(llm_messages, tool_messages) == "weather_tool"

    def test_check_exit_conditions_errored_exit_tool_cancels_a_succeeding_one(self, weather_tool):
        """An errored exit-condition tool cancels the exit even when another exit-condition tool succeeded."""
        agent = Agent(
            chat_generator=MockChatGenerator("Hello"), tools=[weather_tool], exit_conditions=["weather_tool", "search"]
        )

        ok_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        errored_call = ToolCall(tool_name="search", arguments={"q": "weather Berlin"})

        # The succeeding exit tool is listed first, so a naive first-match scan would wrongly return it.
        llm_messages = [ChatMessage.from_assistant(tool_calls=[ok_call, errored_call])]
        tool_messages = [
            ChatMessage.from_tool(tool_result="ok", origin=ok_call, error=False),
            ChatMessage.from_tool(tool_result="boom", origin=errored_call, error=True),
        ]

        assert agent._check_exit_conditions(llm_messages, tool_messages) is None

    def test_exit_condition_exits(self, weather_tool):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(
            chat_generator=MockChatGenerator(tool_call_message), tools=[weather_tool], exit_conditions=["weather_tool"]
        )

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
        # The exit reason is the tool that triggered the exit, and `last_message` is that tool's result.
        assert result["exit_reason"] == "weather_tool"

    def test_exit_condition_on_tool_provided_at_runtime(self, weather_tool):
        """An exit condition naming a tool absent at init still triggers once that tool is provided at runtime."""
        # weather_tool is NOT among the init tools, but it is named as an exit condition. The model calls
        # weather_tool, which is supplied only at runtime.
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(
            chat_generator=MockChatGenerator(tool_call_message),
            tools=[],
            exit_conditions=["weather_tool"],
            max_agent_steps=5,
        )

        result = agent.run([ChatMessage.from_user("What's the weather in Berlin?")], tools=[weather_tool])

        # The agent exits right after the exit-condition tool runs (single step), not at max_agent_steps.
        assert result["step_count"] == 1
        assert result["messages"][-2].tool_call.tool_name == "weather_tool"
        assert (
            result["messages"][-1].tool_call_result.result
            == '{"weather": "mostly sunny", "temperature": 7, "unit": "celsius"}'
        )
        assert result["messages"][-1] == result["last_message"]

    def test_does_not_exit_on_empty_assistant_message(self, weather_tool):
        # The first reply simulates the LLM producing an invalid tool call that our code discards, leaving an
        # assistant message with empty text and no tool calls. This must not be treated as a "text" exit
        # condition, so the agent keeps looping and recovers on the second reply.
        replies = [ChatMessage.from_assistant(text=""), "The weather is sunny."]
        agent = Agent(chat_generator=MockChatGenerator(replies), tools=[weather_tool], exit_conditions=["text"])

        result = agent.run([ChatMessage.from_user("What's the weather?")])

        assert result["step_count"] == 2
        assert result["last_message"].text == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_does_not_exit_on_empty_assistant_message_async(self, weather_tool):
        replies = [ChatMessage.from_assistant(text=""), "The weather is sunny."]
        agent = Agent(chat_generator=MockChatGenerator(replies), tools=[weather_tool], exit_conditions=["text"])

        result = await agent.run_async([ChatMessage.from_user("What's the weather?")])

        assert result["step_count"] == 2
        assert result["last_message"].text == "The weather is sunny."

    def test_text_exit(self, weather_tool):
        """A plain assistant reply with no tool calls reports the `"text"` exit reason."""
        agent = Agent(chat_generator=MockChatGenerator("Berlin is sunny."), tools=[weather_tool])
        result = agent.run([ChatMessage.from_user("Weather in Berlin?")])
        assert result["exit_reason"] == "text"

    def test_tool_exit_reports_the_first_matching_tool(self, weather_tool, component_tool):
        """When several exit-condition tools are called in one step, the first one encountered is reported."""
        parallel_calls = ChatMessage.from_assistant(
            tool_calls=[
                ToolCall(tool_name="parrot", arguments={"parrot": "hi"}),
                ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ]
        )
        agent = Agent(
            chat_generator=MockChatGenerator(parallel_calls),
            tools=[weather_tool, component_tool],
            exit_conditions=["weather_tool", "parrot"],
        )
        result = agent.run([ChatMessage.from_user("Go")])
        assert result["exit_reason"] == "parrot"

    def test_max_steps_exit(self, weather_tool, caplog):
        """Exhausting `max_agent_steps` before meeting an exit condition reports `"max_agent_steps"`."""
        # The model keeps requesting a (non-exit-condition) tool call, so the loop never exits on its own.
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(chat_generator=MockChatGenerator(tool_call_message), tools=[weather_tool], max_agent_steps=2)

        with caplog.at_level(logging.WARNING):
            result = agent.run([ChatMessage.from_user("Weather in Berlin?")])

        assert "Agent reached maximum agent steps" in caplog.text
        assert result["exit_reason"] == "max_agent_steps"
        assert result["step_count"] == 2

    def test_exit_reason_is_readable_in_after_run_hook(self, weather_tool):
        """The `after_run` hook can read `exit_reason` to, e.g., append a fallback answer when max steps is hit."""
        seen_reasons: list[str] = []

        @hook
        def fallback_on_max_steps(state: State) -> None:
            reason = state.get("exit_reason")
            seen_reasons.append(reason)
            if reason == "max_agent_steps":
                state.set("messages", [ChatMessage.from_assistant("Fallback answer.")])

        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(
            chat_generator=MockChatGenerator(tool_call_message),
            tools=[weather_tool],
            max_agent_steps=2,
            hooks={"after_run": [fallback_on_max_steps]},
        )
        result = agent.run([ChatMessage.from_user("Weather in Berlin?")])
        assert seen_reasons == ["max_agent_steps"]
        assert result["exit_reason"] == "max_agent_steps"
        assert result["last_message"].text == "Fallback answer."

    @pytest.mark.asyncio
    async def test_text_exit_async(self, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator("Berlin is sunny."), tools=[weather_tool])
        result = await agent.run_async([ChatMessage.from_user("Weather in Berlin?")])
        assert result["exit_reason"] == "text"

    @pytest.mark.asyncio
    async def test_tool_exit_reason_is_the_tool_name_async(self, weather_tool):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(
            chat_generator=MockChatGenerator(tool_call_message), tools=[weather_tool], exit_conditions=["weather_tool"]
        )
        result = await agent.run_async([ChatMessage.from_user("Weather in Berlin?")])
        assert result["exit_reason"] == "weather_tool"

    @pytest.mark.asyncio
    async def test_max_steps_exit_async(self, weather_tool):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )
        agent = Agent(chat_generator=MockChatGenerator(tool_call_message), tools=[weather_tool], max_agent_steps=2)
        result = await agent.run_async([ChatMessage.from_user("Weather in Berlin?")])
        assert result["exit_reason"] == "max_agent_steps"


class TestAgentTracing:
    def test_tracing_span_run(self, spying_tracer, weather_tool):
        agent = Agent(chat_generator=MockChatGeneratorWithoutRunAsync(), tools=[weather_tool])

        result = agent.run([ChatMessage.from_user("What's the weather in Paris?")])

        assert [s.operation_name for s in spying_tracer.spans] == [
            "haystack.agent.run",
            "haystack.agent.step",
            "haystack.agent.step.llm",
        ]
        run_span, step_span, llm_span = spying_tracer.spans
        assert step_span.parent_span is run_span
        assert llm_span.parent_span is step_span

        assert run_span.tags == {
            "haystack.agent.max_steps": 100,
            "haystack.agent.tools": [weather_tool],
            "haystack.agent.exit_conditions": ["text"],
            "haystack.agent.state_schema": {
                "messages": {
                    "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                    "handler": "haystack.components.agents.state.state_utils.merge_lists",
                },
                "step_count": {"type": "int", "handler": "haystack.components.agents.state.state_utils.replace_values"},
                "token_usage": {
                    "type": "dict[str, typing.Any]",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
                "tool_call_counts": {
                    "type": "dict[str, int]",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
                "exit_reason": {
                    "type": "str",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
                "continue_run": {
                    "type": "bool",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
                "tools": {"type": "list", "handler": "haystack.components.agents.state.state_utils.replace_values"},
                "hook_context": {
                    "type": "dict[str, typing.Any]",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
                "context_tokens": {
                    "type": "int",
                    "handler": "haystack.components.agents.state.state_utils.replace_values",
                },
            },
            "haystack.agent.input": {
                "messages": [ChatMessage.from_user("What's the weather in Paris?")],
                "streaming_callback": None,
            },
            "haystack.agent.output": result,
            "haystack.agent.steps_taken": 1,
        }

        assert step_span.tags == {"haystack.agent.step": 0}

        assert llm_span.tags == {
            "haystack.agent.step.llm.input": {
                "messages": [ChatMessage.from_user("What's the weather in Paris?")],
                "tools": [weather_tool],
            },
            "haystack.agent.step.llm.output": {"replies": [ChatMessage.from_assistant("Hello")]},
        }

    @pytest.mark.asyncio
    async def test_tracing_span_run_async(self, spying_tracer, weather_tool):
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[weather_tool])

        result = await agent.run_async([ChatMessage.from_user("What's the weather in Paris?")])

        assert [s.operation_name for s in spying_tracer.spans] == [
            "haystack.agent.run",
            "haystack.agent.step",
            "haystack.agent.step.llm",
        ]
        run_span, _, llm_span = spying_tracer.spans
        assert run_span.tags["haystack.agent.steps_taken"] == 1
        assert run_span.tags["haystack.agent.output"] == result
        assert llm_span.tags["haystack.agent.step.llm.output"]["replies"][0].text == "Hello"

    def test_tracing_span_run_reflects_runtime_tools(self, spying_tracer, weather_tool, component_tool):
        """The `haystack.agent.tools` span tag should reflect the tools selected for the run, not just init tools."""
        agent = Agent(chat_generator=MockChatGeneratorWithoutRunAsync(), tools=[weather_tool, component_tool])

        # Override at runtime to only use weather_tool, even though the agent was configured with both.
        agent.run([ChatMessage.from_user("What's the weather in Paris?")], tools=[weather_tool.name])

        run_span = spying_tracer.spans[0]
        assert run_span.tags["haystack.agent.tools"] == [weather_tool]

    def test_tracing_span_run_with_tool_call(self, spying_tracer, weather_tool):
        chat_generator = MockChatGenerator(
            [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
                ),
                "done",
            ]
        )
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])

        agent.run([ChatMessage.from_user("What's the weather in Berlin?")])

        assert [s.operation_name for s in spying_tracer.spans] == [
            "haystack.agent.run",
            "haystack.agent.step",
            "haystack.agent.step.llm",
            "haystack.agent.step.tool",
            "haystack.agent.step",
            "haystack.agent.step.llm",
        ]

        # The single tool call gets its own tool span carrying the tool's identity plus its call args and result.
        tool_span = spying_tracer.spans[3]
        assert tool_span.tags == {
            "haystack.tool.name": "weather_tool",
            "haystack.tool.description": "Provides weather information for a given location.",
            "haystack.agent.step.tool.input": {"location": "Berlin"},
            "haystack.agent.step.tool.output": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        }

        assert spying_tracer.spans[0].tags["haystack.agent.steps_taken"] == 2

    def test_tracing_span_run_with_parallel_tool_calls(self, spying_tracer, weather_tool):
        """Each tool call in a step gets its own `haystack.agent.step.tool` span instead of one grouped span."""
        agent = Agent(chat_generator=_parallel_tool_calling_generator(), tools=[weather_tool])

        agent.run([ChatMessage.from_user("What's the weather in Berlin and Paris?")])

        tool_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.agent.step.tool"]
        # Two tool calls -> two tool spans, one per call, each carrying its own identity, arguments, and result.
        assert len(tool_spans) == 2
        for span in tool_spans:
            assert span.tags["haystack.tool.name"] == "weather_tool"
        assert {span.tags["haystack.agent.step.tool.input"]["location"] for span in tool_spans} == {"Berlin", "Paris"}

    @pytest.mark.asyncio
    async def test_tracing_span_run_async_with_parallel_tool_calls(self, spying_tracer, weather_tool):
        """The async path also emits one `haystack.agent.step.tool` span per tool call."""
        agent = Agent(chat_generator=_parallel_tool_calling_generator(), tools=[weather_tool])

        await agent.run_async([ChatMessage.from_user("What's the weather in Berlin and Paris?")])

        tool_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.agent.step.tool"]
        assert len(tool_spans) == 2
        assert {span.tags["haystack.agent.step.tool.input"]["location"] for span in tool_spans} == {"Berlin", "Paris"}

    def test_tracing_in_pipeline(self, spying_tracer, weather_tool):
        agent = Agent(chat_generator=MockChatGeneratorWithoutRunAsync(), tools=[weather_tool])

        pipeline = Pipeline()
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("Hello {{location}}")])
        )
        pipeline.add_component("agent", agent)
        pipeline.connect("prompt_builder.prompt", "agent.messages")

        pipeline.run(data={"prompt_builder": {"location": "Berlin"}})

        assert [s.operation_name for s in spying_tracer.spans] == [
            "haystack.pipeline.run",
            "haystack.component.run",
            "haystack.component.run",
            "haystack.agent.run",
            "haystack.agent.step",
            "haystack.agent.step.llm",
        ]
        component_names = [
            s.tags["haystack.component.name"]
            for s in spying_tracer.spans
            if s.operation_name == "haystack.component.run"
        ]
        assert component_names == ["prompt_builder", "agent"]

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
    @staticmethod
    def _agent_with_duplicate_tool_names() -> Agent:
        def make_tool(description: str) -> Tool:
            return Tool(
                name="same_name",
                description=description,
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            )

        agent = Agent(
            chat_generator=MockChatGenerator("Hello"),
            tools=[Toolset([make_tool("first")]), Toolset([make_tool("second")])],
        )
        agent.warm_up()
        return agent

    def test_run_raises_on_duplicate_tool_names_across_toolsets(self):
        agent = self._agent_with_duplicate_tool_names()
        with pytest.raises(ValueError, match="Duplicate tool names"):
            agent.run(messages=[ChatMessage.from_user("hi")])

    @pytest.mark.asyncio
    async def test_run_async_raises_on_duplicate_tool_names_across_toolsets(self):
        agent = self._agent_with_duplicate_tool_names()
        with pytest.raises(ValueError, match="Duplicate tool names"):
            await agent.run_async(messages=[ChatMessage.from_user("hi")])

    def test_tool_selection_new_tool(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator("Hello")
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")
        result = agent._select_tools([component_tool])
        assert result == [component_tool]

    def test_tool_selection_existing_tools(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator("Hello")
        agent = Agent(
            chat_generator=chat_generator,
            tools=[weather_tool, component_tool],
            system_prompt="This is a system prompt.",
        )
        result = agent._select_tools(None)
        assert result == [weather_tool, component_tool]

    def test_tool_selection_invalid_type(self, weather_tool: Tool, component_tool: Tool):
        chat_generator = MockChatGenerator("Hello")
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


class TestRegisterPromptVariables:
    def test_register_prompt_variables_warning_when_no_prompt_and_required_variables(self, make_agent, caplog):
        make_agent(required_variables=["name"])
        assert "The parameter required_variables is provided but neither" in caplog.text

    def test_register_prompt_variables_no_warning_when_no_prompt_and_default(self, make_agent, caplog):
        make_agent()
        assert "The parameter required_variables is provided but neither" not in caplog.text

    def test_register_prompt_variables_all_required_by_default(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("Question: {{question}}"))
        assert agent._user_chat_prompt_builder.required_variables == "*"

        socket = agent.__haystack_input__._sockets_dict["question"]
        assert socket.is_mandatory

    def test_register_prompt_variables_all_optional_with_none(self, make_agent):
        agent = make_agent(user_prompt=_user_msg("Question: {{question}}"), required_variables=None)

        socket = agent.__haystack_input__._sockets_dict["question"]
        assert not socket.is_mandatory

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

    def test_prompt_wrong_role_raises_at_init(self, make_agent):
        with pytest.raises(ValueError, match="system_prompt message block must have role 'system'"):
            make_agent(system_prompt=_user_msg("This is a user message, not system."))
        with pytest.raises(ValueError, match="user_prompt message block must have role 'user'"):
            make_agent(user_prompt=_sys_msg("This is a system message, not user."))

    def test_dynamic_prompt_role_raises_at_runtime(self, make_agent):
        agent = make_agent(user_prompt="{% message role=role_name %}Q: {{question}}{% endmessage %}")
        with pytest.raises(ValueError, match="user_prompt must render to a user message"):
            agent.run(messages=[], role_name="assistant", question="Will it snow?")

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

    def test_user_prompt_plain_string_with_template_variables(self, make_agent):
        agent = make_agent(user_prompt="Question: {{question}}")
        result = agent.run(messages=[], question="Will it snow?")
        user_messages = [m for m in result["messages"] if m.is_from(ChatRole.USER)]
        assert user_messages[0].text == "Question: Will it snow?"

        input_names = set(agent.__haystack_input__._sockets_dict.keys())
        assert "question" in input_names

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
        chat_generator = MockChatGenerator("Hello")

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
                    return {}  # _NoOutputProduced → blocks AttachmentsBuilder
                return {"processed_files": files}

        @component
        class AttachmentsBuilder:
            """Builds attachment messages; mandatory processed_files from FilesProcessor."""

            @component.output_types(prompt=list[ChatMessage])
            def run(self, processed_files: list[str]) -> dict:
                return {"prompt": [ChatMessage.from_user(f"Files: {processed_files}")]}

        chat_generator = MockChatGenerator("Hello")
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

    def test_warm_up_multiple_tools(self):
        tool1 = self._make_tracking_tool("tool1")
        tool2 = self._make_tracking_tool("tool2")
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[tool1, tool2])

        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up
        agent.warm_up()
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up

    def test_warm_up_toolset(self):
        inner_tool = self._make_tracking_tool()
        toolset = self._make_tracking_toolset([inner_tool])
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=toolset)

        assert not toolset.was_warmed_up
        agent.warm_up()
        assert toolset.was_warmed_up

    def test_warm_up_mixed_list_of_tools_and_toolsets(self):
        tool1 = self._make_tracking_tool("standalone_tool1")
        tool2 = self._make_tracking_tool("standalone_tool2")
        tool3 = self._make_tracking_tool("toolset_tool1")
        toolset1 = self._make_tracking_toolset([tool3])
        tool4 = self._make_tracking_tool("toolset_tool2")
        toolset2 = self._make_tracking_toolset([tool4])

        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[tool1, toolset1, tool2, toolset2])

        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up
        assert not toolset1.was_warmed_up
        assert not toolset2.was_warmed_up
        agent.warm_up()
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up

    def test_warm_up_rewarms_tools_on_every_call(self):
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

        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[tool])
        agent.warm_up()
        agent.warm_up()
        agent.warm_up()

        assert call_count["n"] == 3

    @pytest.mark.parametrize(
        "initial_tools",
        [
            pytest.param([], id="empty"),
            pytest.param(
                [
                    Tool(
                        name="mcp_not_connected_placeholder_123",
                        description="Placeholder tool before connection",
                        parameters={"type": "object", "properties": {}},
                        function=lambda: "placeholder",
                    )
                ],
                id="placeholder",
            ),
        ],
    )
    def test_warm_up_loads_lazy_toolset(self, initial_tools):
        # Before warm_up(), a lazy toolset (e.g. MCPToolset) is either empty or contains a placeholder tool.
        # Agent.warm_up() must load the real tools in both cases.
        actual_tool = Tool(
            name="get_time",
            description="Get the current time in ISO format",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: "2024-12-01T12:00:00Z",
        )

        class LazyToolset(Toolset):
            def __init__(self):
                self._connected = False
                super().__init__(list(initial_tools))

            def warm_up(self):
                if not self._connected:
                    self.tools = [actual_tool]
                    self._connected = True

        toolset = LazyToolset()
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=toolset)
        assert toolset.tools == initial_tools
        agent.warm_up()
        assert toolset.tools == [actual_tool]

    def test_run_warms_lazy_toolset_before_tool_selection(self):
        """
        Agent.run() must warm up lazy toolsets before passing tools to the ChatGenerator and before executing tool calls
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

    def test_run_warms_up_per_run_toolset(self):
        """Per-run tools passed to run() are not covered by Agent.warm_up() and must be warmed up at run time."""
        init_tool = self._make_tracking_tool("init_tool")
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=Toolset([init_tool]))
        agent.warm_up()

        per_run_tool = self._make_tracking_tool("per_run_tool")
        per_run_toolset = self._make_tracking_toolset([per_run_tool])
        assert not per_run_toolset.was_warmed_up
        assert not per_run_tool.was_warmed_up

        agent.run(messages=[ChatMessage.from_user("hi")], tools=per_run_toolset)

        assert per_run_toolset.was_warmed_up
        assert per_run_tool.was_warmed_up

    def test_run_warms_up_per_run_list_of_tools_and_toolsets(self):
        """A per-run list of Tools and Toolsets must be warmed up at run time."""
        init_tool = self._make_tracking_tool("init_tool")
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=[init_tool])
        agent.warm_up()

        per_run_tool = self._make_tracking_tool("per_run_tool")
        toolset_tool = self._make_tracking_tool("toolset_tool")
        per_run_toolset = self._make_tracking_toolset([toolset_tool])

        agent.run(messages=[ChatMessage.from_user("hi")], tools=[per_run_tool, per_run_toolset])

        assert per_run_tool.was_warmed_up
        assert per_run_toolset.was_warmed_up
        assert toolset_tool.was_warmed_up

    @pytest.mark.asyncio
    async def test_run_async_warms_up_per_run_toolset(self):
        """The async run path must also warm up per-run tools."""
        init_tool = self._make_tracking_tool("init_tool")
        agent = Agent(chat_generator=MockChatGenerator("Hello"), tools=Toolset([init_tool]))
        agent.warm_up()

        per_run_tool = self._make_tracking_tool("per_run_tool")
        per_run_toolset = self._make_tracking_toolset([per_run_tool])

        await agent.run_async(messages=[ChatMessage.from_user("hi")], tools=per_run_toolset)

        assert per_run_toolset.was_warmed_up
        assert per_run_tool.was_warmed_up


class TestComponentLifecycle:
    def test_warm_up_delegates_to_chat_generator(self, weather_tool):
        chat_generator = MockChatGenerator("Hello")
        chat_generator.warm_up = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[weather_tool], system_prompt="This is a system prompt.")

        agent.warm_up()
        chat_generator.warm_up.assert_called_once()

        chat_generator.warm_up.reset_mock()
        agent.run([ChatMessage.from_user("What is the weather in Berlin?")])
        # warm_up runs twice here: the Agent delegates to the generator, and the generator's own run() self-warms
        assert chat_generator.warm_up.call_count == 2

    @pytest.mark.asyncio
    async def test_warm_up_async_delegates_to_chat_generator(self):
        chat_generator = MockChatGenerator("Hello")
        chat_generator.warm_up_async = AsyncMock()
        chat_generator.warm_up = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[])
        await agent.warm_up_async()
        chat_generator.warm_up_async.assert_awaited_once()
        chat_generator.warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_warm_up_async_falls_back_to_sync_warm_up(self):
        chat_generator = MockChatGeneratorWithoutRunAsync()
        chat_generator.warm_up = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[])
        await agent.warm_up_async()
        chat_generator.warm_up.assert_called_once()

    def test_close_delegates_to_chat_generator(self):
        chat_generator = MockChatGenerator("Hello")
        chat_generator.close = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[])
        agent.close()
        chat_generator.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async_delegates_to_chat_generator(self):
        chat_generator = MockChatGenerator("Hello")
        chat_generator.close_async = AsyncMock()
        agent = Agent(chat_generator=chat_generator, tools=[])
        await agent.close_async()
        chat_generator.close_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_async_falls_back_to_sync_close(self):
        chat_generator = MockChatGenerator("Hello")
        chat_generator.close = MagicMock()
        agent = Agent(chat_generator=chat_generator, tools=[])
        await agent.close_async()
        chat_generator.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifecycle_is_safe_when_chat_generator_lacks_methods(self):
        agent = Agent(chat_generator=MockChatGeneratorWithoutRunAsync(), tools=[])
        agent.warm_up()
        await agent.warm_up_async()
        agent.close()
        await agent.close_async()


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

        chat_generator = MockChatGenerator("Hello")
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
