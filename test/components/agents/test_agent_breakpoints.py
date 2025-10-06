# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Optional, Union

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, Breakpoint, ToolBreakpoint
from haystack.tools import Tool, Toolset


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


@component
class MockChatGenerator:
    def __init__(self):
        self._counter = 0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "MockChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGenerator":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(
        self, messages: list[ChatMessage], tools: Optional[Union[list[Tool], Toolset]] = None, **kwargs
    ) -> dict[str, Any]:
        if self._counter == 0:
            self._counter += 1
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I'll help you check the weather.",
                        tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                    )
                ]
            }
        else:
            return {"replies": [ChatMessage.from_assistant("The weather in Berlin is sunny.")]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: Optional[Union[list[Tool], Toolset]] = None, **kwargs
    ) -> dict[str, Any]:
        if self._counter == 0:
            self._counter += 1
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I'll help you check the weather.",
                        tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})],
                    )
                ]
            }
        else:
            return {"replies": [ChatMessage.from_assistant("The weather in Berlin is sunny.")]}


@pytest.fixture
def agent(weather_tool):
    return Agent(chat_generator=MockChatGenerator(), tools=[weather_tool])


@pytest.fixture
def chat_generator_serialization_schema():
    return {
        "type": "object",
        "properties": {
            "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
            "tools": {"type": "array", "items": {"type": "haystack.tools.tool.Tool"}},
        },
    }


class TestAgentBreakpoints:
    def test_run_with_chat_generator_breakpoint(self, agent, chat_generator_serialization_schema):
        agent_breakpoint = AgentBreakpoint(
            break_point=Breakpoint(component_name="chat_generator"), agent_name="test_agent"
        )
        with pytest.raises(BreakpointException) as exc_info:
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)
        assert isinstance(exc_info.value, BreakpointException)
        assert exc_info.value.component == "chat_generator"
        assert exc_info.value.inputs == {
            "chat_generator": {
                "serialization_schema": chat_generator_serialization_schema,
                "serialized_data": {
                    "messages": [
                        {
                            "role": "user",
                            "meta": {},
                            "name": None,
                            "content": [{"text": "What's the weather in Berlin?"}],
                        }
                    ],
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
                                "function": "test_agent_breakpoints.weather_function",
                                "outputs_to_string": None,
                                "inputs_from_state": None,
                                "outputs_to_state": None,
                            },
                        }
                    ],
                },
            },
            "tool_invoker": {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {"type": "array", "items": {}},
                        "state": {"type": "haystack.components.agents.state.state.State"},
                        "tools": {"type": "array", "items": {"type": "haystack.tools.tool.Tool"}},
                    },
                },
                "serialized_data": {
                    "messages": [],
                    "state": {
                        "schema": {
                            "messages": {
                                "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                                "handler": "haystack.components.agents.state.state_utils.merge_lists",
                            }
                        },
                        "data": {
                            "serialization_schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                                    }
                                },
                            },
                            "serialized_data": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "meta": {},
                                        "name": None,
                                        "content": [{"text": "What's the weather in Berlin?"}],
                                    }
                                ]
                            },
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
                                "function": "test_agent_breakpoints.weather_function",
                                "outputs_to_string": None,
                                "inputs_from_state": None,
                                "outputs_to_state": None,
                            },
                        }
                    ],
                },
            },
        }
        assert exc_info.value.results == {
            "schema": {
                "messages": {
                    "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                    "handler": "haystack.components.agents.state.state_utils.merge_lists",
                }
            },
            "data": {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                        }
                    },
                },
                "serialized_data": {
                    "messages": [
                        {
                            "role": "user",
                            "meta": {},
                            "name": None,
                            "content": [{"text": "What's the weather in Berlin?"}],
                        }
                    ]
                },
            },
        }

    def test_run_with_tool_invoker_breakpoint(self, agent, chat_generator_serialization_schema):
        agent_breakpoint = AgentBreakpoint(
            break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool"), agent_name="test_agent"
        )
        with pytest.raises(BreakpointException) as exc_info:
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)

        assert isinstance(exc_info.value, BreakpointException)
        assert exc_info.value.component == "tool_invoker"
        assert exc_info.value.inputs == {
            "chat_generator": {
                "serialization_schema": chat_generator_serialization_schema,
                "serialized_data": {
                    "messages": [
                        {
                            "role": "user",
                            "meta": {},
                            "name": None,
                            "content": [{"text": "What's the weather in Berlin?"}],
                        }
                    ],
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
                                "function": "test_agent_breakpoints.weather_function",
                                "outputs_to_string": None,
                                "inputs_from_state": None,
                                "outputs_to_state": None,
                            },
                        }
                    ],
                },
            },
            "tool_invoker": {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                        },
                        "state": {"type": "haystack.components.agents.state.state.State"},
                        "tools": {"type": "array", "items": {"type": "haystack.tools.tool.Tool"}},
                    },
                },
                "serialized_data": {
                    "messages": [
                        {
                            "role": "assistant",
                            "meta": {},
                            "name": None,
                            "content": [
                                {"text": "I'll help you check the weather."},
                                {
                                    "tool_call": {
                                        "tool_name": "weather_tool",
                                        "arguments": {"location": "Berlin"},
                                        "id": None,
                                    }
                                },
                            ],
                        }
                    ],
                    "state": {
                        "schema": {
                            "messages": {
                                "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                                "handler": "haystack.components.agents.state.state_utils.merge_lists",
                            }
                        },
                        "data": {
                            "serialization_schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                                    }
                                },
                            },
                            "serialized_data": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "meta": {},
                                        "name": None,
                                        "content": [{"text": "What's the weather in Berlin?"}],
                                    },
                                    {
                                        "role": "assistant",
                                        "meta": {},
                                        "name": None,
                                        "content": [
                                            {"text": "I'll help you check the weather."},
                                            {
                                                "tool_call": {
                                                    "tool_name": "weather_tool",
                                                    "arguments": {"location": "Berlin"},
                                                    "id": None,
                                                }
                                            },
                                        ],
                                    },
                                ]
                            },
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
                                "function": "test_agent_breakpoints.weather_function",
                                "outputs_to_string": None,
                                "inputs_from_state": None,
                                "outputs_to_state": None,
                            },
                        }
                    ],
                },
            },
        }
        assert exc_info.value.results == {
            "schema": {
                "messages": {
                    "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                    "handler": "haystack.components.agents.state.state_utils.merge_lists",
                }
            },
            "data": {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                        }
                    },
                },
                "serialized_data": {
                    "messages": [
                        {
                            "role": "user",
                            "meta": {},
                            "name": None,
                            "content": [{"text": "What's the weather in Berlin?"}],
                        },
                        {
                            "role": "assistant",
                            "meta": {},
                            "name": None,
                            "content": [
                                {"text": "I'll help you check the weather."},
                                {
                                    "tool_call": {
                                        "tool_name": "weather_tool",
                                        "arguments": {"location": "Berlin"},
                                        "id": None,
                                    }
                                },
                            ],
                        },
                    ]
                },
            },
        }

    def test_resume_from_chat_generator(self, agent, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        agent_breakpoint = AgentBreakpoint(
            break_point=Breakpoint(component_name="chat_generator", snapshot_file_path=debug_path),
            agent_name="test_agent",
        )

        try:
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_chat_generator_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = agent.run(
            messages=[ChatMessage.from_user("This is actually ignored when resuming from snapshot.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        # There should be 4 messages: user + assistant + tool call result + final assistant message
        assert len(result["messages"]) == 4

    def test_resume_from_tool_invoker(self, agent, tmp_path):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        debug_path = str(tmp_path / "debug_snapshots")
        tool_bp = ToolBreakpoint(component_name="tool_invoker", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")

        try:
            agent.run(messages=messages, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = agent.run(
            messages=[ChatMessage.from_user("This is actually ignored when resuming from snapshot.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    def test_invalid_combination_breakpoint_and_pipeline_snapshot(self, agent):
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
        with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
            agent.run(
                messages=messages,
                break_point=agent_breakpoint,
                snapshot=AgentSnapshot(component_inputs={}, component_visits={}, break_point=agent_breakpoint),
            )

    def test_breakpoint_with_invalid_component_name(self):
        invalid_bp = Breakpoint(component_name="invalid_breakpoint")
        with pytest.raises(ValueError):
            AgentBreakpoint(break_point=invalid_bp, agent_name="test_agent")

    def test_breakpoint_with_invalid_tool_name(self, agent):
        with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
            agent_breakpoint = AgentBreakpoint(
                break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="invalid_tool"),
                agent_name="test_agent",
            )
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_live_resume_from_tool_invoker(self, tmp_path, weather_tool):
        agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o"), tools=[weather_tool])
        debug_path = str(tmp_path / "debug_snapshots")
        agent_breakpoint = AgentBreakpoint(
            break_point=ToolBreakpoint(component_name="tool_invoker", snapshot_file_path=debug_path),
            agent_name="test_agent",
        )

        try:
            agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = agent.run(
            messages=[ChatMessage.from_user("This is actually ignored when resuming from snapshot.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) == 4
        assert "berlin" in result["last_message"].text.lower()


class TestAsyncAgentBreakpoints:
    @pytest.mark.asyncio
    async def test_run_async_with_chat_generator_breakpoint(self, agent):
        agent_breakpoint = AgentBreakpoint(
            break_point=Breakpoint(component_name="chat_generator"), agent_name="test_agent"
        )
        with pytest.raises(BreakpointException) as exc_info:
            await agent.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
            )
        assert exc_info.value.component == "chat_generator"
        assert "messages" in exc_info.value.inputs["chat_generator"]["serialized_data"]

    @pytest.mark.asyncio
    async def test_run_async_with_tool_invoker_breakpoint(self, agent):
        agent_breakpoint = AgentBreakpoint(
            break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool"), agent_name="test"
        )
        with pytest.raises(BreakpointException) as exc_info:
            await agent.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
            )

        assert exc_info.value.component == "tool_invoker"
        assert "messages" in exc_info.value.inputs["tool_invoker"]["serialized_data"]

    @pytest.mark.asyncio
    async def test_resume_from_chat_generator_async(self, agent, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        chat_generator_bp = Breakpoint(component_name="chat_generator", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")

        try:
            await agent.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
            )
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_chat_generator_*.json"))
        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = await agent.run_async(
            messages=[ChatMessage.from_user("This is actually ignored when resuming from snapshot.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) == 4

    @pytest.mark.asyncio
    async def test_resume_from_tool_invoker_async(self, agent, tmp_path):
        debug_path = str(tmp_path / "debug_snapshots")
        messages = [ChatMessage.from_user("What's the weather in Berlin?")]
        tool_bp = ToolBreakpoint(component_name="tool_invoker", tool_name="weather_tool", snapshot_file_path=debug_path)
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")

        try:
            await agent.run_async(messages=messages, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        snapshot_files = list(Path(debug_path).glob("test_agent_tool_invoker_*.json"))

        assert len(snapshot_files) > 0
        latest_snapshot_file = str(max(snapshot_files, key=os.path.getctime))

        result = await agent.run_async(
            messages=[ChatMessage.from_user("This is actually ignored when resuming from snapshot.")],
            snapshot=load_pipeline_snapshot(latest_snapshot_file).agent_snapshot,
        )

        assert "messages" in result
        assert "last_message" in result
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_combination_breakpoint_and_pipeline_snapshot_async(self, agent):
        tool_bp = ToolBreakpoint(component_name="tool_invoker", visit_count=0, tool_name="weather_tool")
        agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test")
        with pytest.raises(ValueError, match="break_point and snapshot cannot be provided at the same time"):
            await agent.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")],
                break_point=agent_breakpoint,
                snapshot=AgentSnapshot(component_inputs={}, component_visits={}, break_point=agent_breakpoint),
            )

    @pytest.mark.asyncio
    async def test_breakpoint_with_invalid_tool_name_async(self, agent):
        agent_breakpoint = AgentBreakpoint(
            break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="invalid_tool"), agent_name="test"
        )
        with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
            await agent.run_async(
                messages=[ChatMessage.from_user("What's the weather in Berlin?")], break_point=agent_breakpoint
            )
