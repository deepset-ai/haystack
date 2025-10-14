# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from haystack import Pipeline
from haystack.components.agents.state import State
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools.tool_invoker import (
    StringConversionError,
    ToolInvoker,
    ToolNotFoundException,
    ToolOutputMergeError,
)
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall, ToolCallResult
from haystack.tools import ComponentTool, Tool, Toolset
from haystack.tools.errors import ToolInvocationError


def weather_function(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


weather_parameters = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters=weather_parameters,
        function=weather_function,
    )


@pytest.fixture
def weather_tool_with_outputs_to_state():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters=weather_parameters,
        function=weather_function,
        outputs_to_state={"weather": {"source": "weather"}},
    )


@pytest.fixture
def faulty_tool():
    def faulty_tool_func(location):
        raise Exception("This tool always fails.")

    faulty_tool_parameters = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }

    return Tool(
        name="faulty_tool",
        description="A tool that always fails when invoked.",
        parameters=faulty_tool_parameters,
        function=faulty_tool_func,
    )


def add_function(num1: int, num2: int):
    return num1 + num2


@pytest.fixture
def tool_set():
    return Toolset(
        tools=[
            Tool(
                name="weather_tool",
                description="Provides weather information for a given location.",
                parameters=weather_parameters,
                function=weather_function,
            ),
            Tool(
                name="addition_tool",
                description="A tool that adds two numbers.",
                parameters={
                    "type": "object",
                    "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                    "required": ["num1", "num2"],
                },
                function=add_function,
            ),
        ]
    )


@pytest.fixture
def invoker(weather_tool):
    return ToolInvoker(tools=[weather_tool], raise_on_failure=True, convert_result_to_json_string=False)


@pytest.fixture
def faulty_invoker(faulty_tool):
    return ToolInvoker(tools=[faulty_tool], raise_on_failure=True, convert_result_to_json_string=False)


class WarmupTrackingTool(Tool):
    """A tool that tracks whether warm_up was called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.was_warmed_up = False

    def warm_up(self):
        self.was_warmed_up = True


class WarmupTrackingToolset(Toolset):
    """A toolset that tracks whether warm_up was called."""

    def __init__(self, tools):
        super().__init__(tools)
        self.was_warmed_up = False

    def warm_up(self):
        self.was_warmed_up = True


class TestToolInvokerCore:
    def test_init(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])

        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {"weather_tool": weather_tool}
        assert invoker.raise_on_failure
        assert not invoker.convert_result_to_json_string

    def test_validate_and_prepare_tools(self, weather_tool, faulty_tool):
        result = ToolInvoker._validate_and_prepare_tools([weather_tool, faulty_tool])
        assert result == {"weather_tool": weather_tool, "faulty_tool": faulty_tool}

        toolset = Toolset([weather_tool, faulty_tool])
        result = ToolInvoker._validate_and_prepare_tools(toolset)
        assert result == {"weather_tool": weather_tool, "faulty_tool": faulty_tool}

    def test_validate_and_prepare_tools_validation_failures(self, weather_tool):
        with pytest.raises(ValueError):
            ToolInvoker._validate_and_prepare_tools([])

        with pytest.raises(ValueError):
            ToolInvoker._validate_and_prepare_tools([weather_tool, weather_tool])

    def test_inject_state_args_no_tool_inputs(self, invoker):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = invoker._inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_no_tool_inputs_component_tool(self, invoker):
        comp = PromptBuilder(template="Hello, {{name}}!")
        prompt_tool = ComponentTool(
            component=comp, name="prompt_tool", description="Creates a personalized greeting prompt."
        )
        state = State(schema={"name": {"type": str}}, data={"name": "James"})
        args = invoker._inject_state_args(tool=prompt_tool, llm_args={}, state=state)
        assert args == {"name": "James"}

    def test_inject_state_args_with_tool_inputs(self, invoker):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            inputs_from_state={"loc": "location"},
        )
        state = State(schema={"location": {"type": str}}, data={"loc": "Berlin"})
        args = invoker._inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_param_in_state_and_llm(self, invoker):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = invoker._inject_state_args(tool=weather_tool, llm_args={"location": "Paris"}, state=state)
        assert args == {"location": "Paris"}


class TestToolInvokerSerde:
    def test_to_dict(self, invoker, weather_tool):
        data = invoker.to_dict()
        assert data == {
            "type": "haystack.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
                "enable_streaming_callback_passthrough": False,
                "streaming_callback": None,
                "max_workers": 4,
            },
        }

    def test_to_dict_with_params(self, weather_tool):
        invoker = ToolInvoker(
            tools=[weather_tool],
            raise_on_failure=False,
            convert_result_to_json_string=True,
            enable_streaming_callback_passthrough=True,
            streaming_callback=print_streaming_chunk,
        )

        assert invoker.to_dict() == {
            "type": "haystack.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": False,
                "convert_result_to_json_string": True,
                "enable_streaming_callback_passthrough": True,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "max_workers": 4,
            },
        }

    def test_from_dict(self, weather_tool):
        data = {
            "type": "haystack.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
                "enable_streaming_callback_passthrough": False,
                "streaming_callback": None,
            },
        }
        invoker = ToolInvoker.from_dict(data)
        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {"weather_tool": weather_tool}
        assert invoker.raise_on_failure
        assert not invoker.convert_result_to_json_string
        assert invoker.streaming_callback is None
        assert invoker.enable_streaming_callback_passthrough is False

    def test_from_dict_with_streaming_callback(self, weather_tool):
        data = {
            "type": "haystack.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
                "enable_streaming_callback_passthrough": True,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            },
        }
        invoker = ToolInvoker.from_dict(data)
        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {"weather_tool": weather_tool}
        assert invoker.raise_on_failure
        assert not invoker.convert_result_to_json_string
        assert invoker.streaming_callback == print_streaming_chunk
        assert invoker.enable_streaming_callback_passthrough is True

    def test_serde_in_pipeline(self, invoker, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        pipeline = Pipeline()
        pipeline.add_component("invoker", invoker)
        pipeline.add_component("chatgenerator", OpenAIChatGenerator())
        pipeline.connect("invoker", "chatgenerator")

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "connection_type_validation": True,
            "max_runs_per_component": 100,
            "components": {
                "invoker": {
                    "type": "haystack.components.tools.tool_invoker.ToolInvoker",
                    "init_parameters": {
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
                                    "function": "tools.test_tool_invoker.weather_function",
                                    "outputs_to_string": None,
                                    "inputs_from_state": None,
                                    "outputs_to_state": None,
                                },
                            }
                        ],
                        "raise_on_failure": True,
                        "convert_result_to_json_string": False,
                        "enable_streaming_callback_passthrough": False,
                        "streaming_callback": None,
                        "max_workers": 4,
                    },
                },
                "chatgenerator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "max_retries": None,
                        "timeout": None,
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
            },
            "connections": [{"sender": "invoker.tool_messages", "receiver": "chatgenerator.messages"}],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline


class TestToolInvokerRun:
    def test_run_with_streaming_callback_finish_reason(self, invoker):
        streaming_chunks = []

        def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        # Check that we received streaming chunks
        assert len(streaming_chunks) >= 2  # At least one for tool result and one for finish reason

        # The last chunk should have finish_reason set to "tool_call_results"
        final_chunk = streaming_chunks[-1]
        assert final_chunk.finish_reason == "tool_call_results"
        assert final_chunk.meta["finish_reason"] == "tool_call_results"
        assert final_chunk.content == ""

    @pytest.mark.asyncio
    async def test_run_async_with_streaming_callback_finish_reason(self, weather_tool):
        streaming_chunks = []

        async def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        tool_invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=True, convert_result_to_json_string=False)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = await tool_invoker.run_async(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        # Check that we received streaming chunks
        assert len(streaming_chunks) >= 2  # At least one for tool result and one for finish reason

        # The last chunk should have finish_reason set to "tool_call_results"
        final_chunk = streaming_chunks[-1]
        assert final_chunk.finish_reason == "tool_call_results"
        assert final_chunk.meta["finish_reason"] == "tool_call_results"
        assert final_chunk.content == ""

    def test_enable_streaming_callback_passthrough(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm_tool = ComponentTool(
            component=OpenAIChatGenerator(),
            name="chat_generator_tool",
            description="A tool that generates chat messages using OpenAI's GPT model.",
        )
        invoker = ToolInvoker(
            tools=[llm_tool], enable_streaming_callback_passthrough=True, streaming_callback=print_streaming_chunk
        )
        with patch("haystack.components.generators.chat.OpenAIChatGenerator.run") as mock_run:
            mock_run.return_value = {"replies": [ChatMessage.from_assistant("Hello! How can I help you?")]}
            invoker.run(
                messages=[
                    ChatMessage.from_assistant(
                        tool_calls=[
                            ToolCall(
                                tool_name="chat_generator_tool",
                                arguments={"messages": [{"role": "user", "content": [{"text": "Hello!"}]}]},
                                id="12345",
                            )
                        ]
                    )
                ]
            )
            mock_run.assert_called_once_with(
                messages=[ChatMessage.from_user(text="Hello!")], streaming_callback=print_streaming_chunk
            )

    def test_enable_streaming_callback_passthrough_runtime(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        llm_tool = ComponentTool(
            component=OpenAIChatGenerator(),
            name="chat_generator_tool",
            description="A tool that generates chat messages using OpenAI's GPT model.",
        )
        invoker = ToolInvoker(
            tools=[llm_tool], enable_streaming_callback_passthrough=True, streaming_callback=print_streaming_chunk
        )
        with patch("haystack.components.generators.chat.OpenAIChatGenerator.run") as mock_run:
            mock_run.return_value = {"replies": [ChatMessage.from_assistant("Hello! How can I help you?")]}
            invoker.run(
                messages=[
                    ChatMessage.from_assistant(
                        tool_calls=[
                            ToolCall(
                                tool_name="chat_generator_tool",
                                arguments={"messages": [{"role": "user", "content": [{"text": "Hello!"}]}]},
                                id="12345",
                            )
                        ]
                    )
                ],
                enable_streaming_callback_passthrough=False,
            )
            mock_run.assert_called_once_with(messages=[ChatMessage.from_user(text="Hello!")])

    def test_run_no_messages(self, invoker):
        result = invoker.run(messages=[])
        assert result["tool_messages"] == []

    def test_run_no_tool_calls(self, invoker):
        user_message = ChatMessage.from_user(text="Hello!")
        assistant_message = ChatMessage.from_assistant(text="How can I help you?")

        result = invoker.run(messages=[user_message, assistant_message])
        assert result["tool_messages"] == []

    def test_run_multiple_tool_calls(self, invoker):
        tool_calls = [
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Rome"}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        result = invoker.run(messages=[message])
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 3

        for i, tool_message in enumerate(result["tool_messages"]):
            assert isinstance(tool_message, ChatMessage)
            assert tool_message.is_from(ChatRole.TOOL)

            assert tool_message.tool_call_results
            tool_call_result = tool_message.tool_call_result

            assert isinstance(tool_call_result, ToolCallResult)
            assert not tool_call_result.error
            assert tool_call_result.origin == tool_calls[i]

    def test_run_tool_calls_with_empty_args(self):
        hello_world_tool = Tool(
            name="hello_world",
            description="A tool that returns a greeting.",
            parameters={"type": "object", "properties": {}},
            function=lambda: "Hello, world!",
        )
        invoker = ToolInvoker(tools=[hello_world_tool])

        tool_call = ToolCall(tool_name="hello_world", arguments={})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[tool_call_message])
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        tool_message = result["tool_messages"][0]
        assert isinstance(tool_message, ChatMessage)
        assert tool_message.is_from(ChatRole.TOOL)

        assert tool_message.tool_call_results
        tool_call_result = tool_message.tool_call_result

        assert isinstance(tool_call_result, ToolCallResult)
        assert not tool_call_result.error

        assert tool_call_result.result == "Hello, world!"

    def test_run_with_tools_override(self, weather_tool, faulty_tool):
        """Tests that tools passed to run override the tools passed in init"""
        invoker = ToolInvoker(tools=[faulty_tool])
        assert invoker._tools_with_names == {"faulty_tool": faulty_tool}
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[message], tools=[weather_tool])

        tool_message = result["tool_messages"][0]
        tool_call_result = tool_message.tool_call_result
        assert not tool_call_result.error
        assert tool_call_result.result == str({"weather": "mostly sunny", "temperature": 7, "unit": "celsius"})
        assert tool_call_result.origin == tool_call

    @pytest.mark.asyncio
    async def test_run_async_with_tools_override(self, weather_tool, faulty_tool):
        """Tests that tools passed to run_async override the tools passed in init"""
        invoker = ToolInvoker(tools=[faulty_tool])
        assert invoker._tools_with_names == {"faulty_tool": faulty_tool}
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = await invoker.run_async(messages=[message], tools=[weather_tool])
        tool_message = result["tool_messages"][0]
        tool_call_result = tool_message.tool_call_result
        assert not tool_call_result.error
        assert tool_call_result.result == str({"weather": "mostly sunny", "temperature": 7, "unit": "celsius"})
        assert tool_call_result.origin == tool_call

    def test_parallel_tool_calling_with_state_updates(self):
        """Test that parallel tool execution with state updates works correctly with the state lock."""
        # Create a shared counter variable to simulate a state value that gets updated
        execution_log = []

        def function_1():
            time.sleep(0.01)
            execution_log.append("tool_1_executed")
            return {"counter": 1, "tool_name": "tool_1"}

        def function_2():
            time.sleep(0.01)
            execution_log.append("tool_2_executed")
            return {"counter": 2, "tool_name": "tool_2"}

        def function_3():
            time.sleep(0.01)
            execution_log.append("tool_3_executed")
            return {"counter": 3, "tool_name": "tool_3"}

        # Create tools that all update the same state key
        tool_1 = Tool(
            name="state_tool_1",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_1,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        tool_2 = Tool(
            name="state_tool_2",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_2,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        tool_3 = Tool(
            name="state_tool_3",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_3,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        # Create ToolInvoker with all three tools
        invoker = ToolInvoker(tools=[tool_1, tool_2, tool_3], raise_on_failure=True)

        state = State(schema={"counter": {"type": int}, "last_tool": {"type": str}})
        tool_calls = [
            ToolCall(tool_name="state_tool_1", arguments={}),
            ToolCall(tool_name="state_tool_2", arguments={}),
            ToolCall(tool_name="state_tool_3", arguments={}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)
        _ = invoker.run(messages=[message], state=state)

        # Verify that all three tools were executed
        assert len(execution_log) == 3
        assert "tool_1_executed" in execution_log
        assert "tool_2_executed" in execution_log
        assert "tool_3_executed" in execution_log

        # Verify that the state was updated correctly
        # Due to parallel execution, we can't predict which tool will be the last to update
        assert state.has("counter")
        assert state.has("last_tool")
        assert state.get("counter") in [1, 2, 3]  # Should be one of the tool values
        assert state.get("last_tool") in ["tool_1", "tool_2", "tool_3"]  # Should be one of the tool names

    @pytest.mark.asyncio
    async def test_async_parallel_tool_calling_with_state_updates(self):
        """Test that parallel tool execution with state updates works correctly with the state lock."""
        # Create a shared counter variable to simulate a state value that gets updated
        execution_log = []

        def function_1():
            time.sleep(0.01)
            execution_log.append("tool_1_executed")
            return {"counter": 1, "tool_name": "tool_1"}

        def function_2():
            time.sleep(0.01)
            execution_log.append("tool_2_executed")
            return {"counter": 2, "tool_name": "tool_2"}

        def function_3():
            time.sleep(0.01)
            execution_log.append("tool_3_executed")
            return {"counter": 3, "tool_name": "tool_3"}

        # Create tools that all update the same state key
        tool_1 = Tool(
            name="state_tool_1",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_1,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        tool_2 = Tool(
            name="state_tool_2",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_2,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        tool_3 = Tool(
            name="state_tool_3",
            description="A tool that updates state counter",
            parameters={"type": "object", "properties": {}},
            function=function_3,
            outputs_to_state={"counter": {"source": "counter"}, "last_tool": {"source": "tool_name"}},
        )

        # Create ToolInvoker with all three tools
        invoker = ToolInvoker(tools=[tool_1, tool_2, tool_3], raise_on_failure=True)

        state = State(schema={"counter": {"type": int}, "last_tool": {"type": str}})
        tool_calls = [
            ToolCall(tool_name="state_tool_1", arguments={}),
            ToolCall(tool_name="state_tool_2", arguments={}),
            ToolCall(tool_name="state_tool_3", arguments={}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)
        _ = await invoker.run_async(messages=[message], state=state)

        # Verify that all three tools were executed
        assert len(execution_log) == 3
        assert "tool_1_executed" in execution_log
        assert "tool_2_executed" in execution_log
        assert "tool_3_executed" in execution_log

        # Verify that the state was updated correctly
        # Due to parallel execution, we can't predict which tool will be the last to update
        assert state.has("counter")
        assert state.has("last_tool")
        assert state.get("counter") in [1, 2, 3]  # Should be one of the tool values
        assert state.get("last_tool") in ["tool_1", "tool_2", "tool_3"]  # Should be one of the tool names

    def test_call_invoker_two_subsequent_run_calls(self, invoker: ToolInvoker):
        tool_calls = [
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Rome"}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        # First call
        result_1 = invoker.run(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result_1
        assert len(result_1["tool_messages"]) == 3

        # Second call
        result_2 = invoker.run(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result_2
        assert len(result_2["tool_messages"]) == 3

    @pytest.mark.asyncio
    async def test_call_invoker_two_subsequent_run_async_calls(self, invoker: ToolInvoker):
        tool_calls = [
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Rome"}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        streaming_callback_called = False

        async def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        # First call
        result_1 = await invoker.run_async(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result_1
        assert len(result_1["tool_messages"]) == 3

        # Second call
        result_2 = await invoker.run_async(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result_2
        assert len(result_2["tool_messages"]) == 3


class TestToolInvokerErrorHandling:
    def test_tool_not_found_error(self, invoker):
        tool_call = ToolCall(tool_name="non_existent_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        with pytest.raises(ToolNotFoundException):
            invoker.run(messages=[tool_call_message])

    def test_tool_not_found_does_not_raise_exception(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=False, convert_result_to_json_string=False)

        tool_call = ToolCall(tool_name="non_existent_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[tool_call_message])
        tool_message = result["tool_messages"][0]

        assert tool_message.tool_call_results[0].error
        assert "not found" in tool_message.tool_call_results[0].result

    def test_tool_invocation_error(self, faulty_invoker):
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        with pytest.raises(ToolInvocationError):
            faulty_invoker.run(messages=[tool_call_message])

    def test_tool_invocation_error_does_not_raise_exception(self, faulty_tool):
        faulty_invoker = ToolInvoker(tools=[faulty_tool], raise_on_failure=False, convert_result_to_json_string=False)

        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = faulty_invoker.run(messages=[tool_call_message])
        tool_message = result["tool_messages"][0]
        assert tool_message.tool_call_results[0].error
        assert "Failed to invoke" in tool_message.tool_call_results[0].result

    def test_string_conversion_error(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            # Pass custom handler that will throw an error when trying to convert tool_result
            outputs_to_string={"handler": lambda x: json.dumps(x)},
        )
        invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=True)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        with pytest.raises(StringConversionError):
            invoker._prepare_tool_result_message(result=tool_result, tool_call=tool_call, tool_to_invoke=weather_tool)

    def test_string_conversion_error_does_not_raise_exception(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            # Pass custom handler that will throw an error when trying to convert tool_result
            outputs_to_string={"handler": lambda x: json.dumps(x)},
        )
        invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=False)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        tool_message = invoker._prepare_tool_result_message(
            result=tool_result, tool_call=tool_call, tool_to_invoke=weather_tool
        )

        assert tool_message.tool_call_results[0].error
        assert "Failed to convert" in tool_message.tool_call_results[0].result

    def test_run_state_merge_error_handled_gracefully(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                # Simulate a State error during merging
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        invoker = ToolInvoker(tools=[weather_tool_with_outputs_to_state], raise_on_failure=False)

        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        result = invoker.run(messages=[message], state=state)

        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1
        assert result["tool_messages"][0].tool_call_results[0].error is True
        assert (
            "Failed to merge tool outputs from tool weather_tool into State"
            in result["tool_messages"][0].tool_call_results[0].result
        )

    def test_run_state_merge_error_raises_when_configured(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                # Simulate a State error during merging
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        invoker = ToolInvoker(tools=[weather_tool_with_outputs_to_state], raise_on_failure=True)

        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        with pytest.raises(ToolOutputMergeError, match="Failed to merge"):
            invoker.run(messages=[message], state=state)

    @pytest.mark.asyncio
    async def test_run_async_state_merge_error_handled_gracefully(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                # Simulate a State error during merging
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        invoker = ToolInvoker(tools=[weather_tool_with_outputs_to_state], raise_on_failure=False)

        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        result = await invoker.run_async(messages=[message], state=state)

        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1
        assert result["tool_messages"][0].tool_call_results[0].error is True
        assert (
            "Failed to merge tool outputs from tool weather_tool into State"
            in result["tool_messages"][0].tool_call_results[0].result
        )

    @pytest.mark.asyncio
    async def test_run_async_state_merge_error_raises_when_configured(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                # Simulate a State error during merging
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        invoker = ToolInvoker(tools=[weather_tool_with_outputs_to_state], raise_on_failure=True)

        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        with pytest.raises(ToolOutputMergeError, match="Failed to merge"):
            await invoker.run_async(messages=[message], state=state)


class TestToolInvokerUtilities:
    def test_default_output_to_string_handler_basic_types(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], convert_result_to_json_string=False)

        assert invoker._default_output_to_string_handler("hello") == "hello"
        assert invoker._default_output_to_string_handler(42) == "42"
        assert invoker._default_output_to_string_handler(3.14) == "3.14"
        assert invoker._default_output_to_string_handler(True) == "True"
        assert invoker._default_output_to_string_handler(None) == "None"

        assert invoker._default_output_to_string_handler([1, 2, 3]) == "[1, 2, 3]"
        assert invoker._default_output_to_string_handler({"key": "value"}) == "{'key': 'value'}"

    def test_default_output_to_string_handler_json_string_mode(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], convert_result_to_json_string=True)

        assert invoker._default_output_to_string_handler("hello") == '"hello"'
        assert invoker._default_output_to_string_handler(42) == "42"
        assert invoker._default_output_to_string_handler(True) == "true"
        assert invoker._default_output_to_string_handler(None) == "null"

        assert invoker._default_output_to_string_handler([1, 2, 3]) == "[1, 2, 3]"
        assert invoker._default_output_to_string_handler({"key": "value"}) == '{"key": "value"}'

        assert invoker._default_output_to_string_handler("Hello 🌍") == '"Hello 🌍"'

    def test_default_output_to_string_handler_with_serializable_objects(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], convert_result_to_json_string=False)

        # Create a mock object with to_dict method
        class MockObject:
            def __init__(self, value):
                self.value = value

            def to_dict(self):
                return {"value": self.value}

        mock_obj = MockObject("test_value")
        result = invoker._default_output_to_string_handler(mock_obj)

        # Should convert to string representation of the dict
        assert "test_value" in result
        assert "value" in result

    def test_merge_tool_outputs_result_not_a_dict(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"weather": {"type": str}})
        invoker._merge_tool_outputs(tool=weather_tool, result="test", state=state)
        assert state.data == {}

    def test_merge_tool_outputs_empty_dict(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"weather": {"type": str}})
        invoker._merge_tool_outputs(tool=weather_tool, result={}, state=state)
        assert state.data == {}

    def test_merge_tool_outputs_no_output_mapping(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"weather": {"type": str}})
        invoker._merge_tool_outputs(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {}

    def test_merge_tool_outputs_with_output_mapping(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_state={"weather": {"source": "weather"}},
        )
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"weather": {"type": str}})
        invoker._merge_tool_outputs(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {"weather": "sunny"}

    def test_merge_tool_outputs_with_output_mapping_2(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_state={"all_weather_results": {}},
        )
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"all_weather_results": {"type": str}})
        invoker._merge_tool_outputs(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {"all_weather_results": {"weather": "sunny", "temperature": 14, "unit": "celsius"}}

    def test_merge_tool_outputs_with_output_mapping_and_handler(self):
        handler = lambda old, new: f"{new}"
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_state={"temperature": {"source": "temperature", "handler": handler}},
        )
        invoker = ToolInvoker(tools=[weather_tool])
        state = State(schema={"temperature": {"type": str}})
        invoker._merge_tool_outputs(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {"temperature": "14"}


class TestWarmUpTools:
    """Tests for Tool/Toolset warm_up through ToolInvoker"""

    def test_tool_invoker_warm_up_with_single_tool(self):
        """Test that ToolInvoker.warm_up() calls warm_up on a single tool."""
        tool = WarmupTrackingTool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test",
        )

        invoker = ToolInvoker(tools=[tool])

        assert not tool.was_warmed_up
        invoker.warm_up()
        assert tool.was_warmed_up

    def test_tool_invoker_warm_up_with_multiple_tools(self):
        """Test that ToolInvoker.warm_up() calls warm_up on multiple tools."""
        tool1 = WarmupTrackingTool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "tool1",
        )
        tool2 = WarmupTrackingTool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "tool2",
        )

        invoker = ToolInvoker(tools=[tool1, tool2])

        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up

        invoker.warm_up()

        assert tool1.was_warmed_up
        assert tool2.was_warmed_up

    def test_tool_invoker_warm_up_with_toolset(self, weather_tool):
        """Test that ToolInvoker.warm_up() calls warm_up on the toolset."""
        toolset = WarmupTrackingToolset([weather_tool])
        invoker = ToolInvoker(tools=toolset)

        assert not toolset.was_warmed_up
        invoker.warm_up()
        assert toolset.was_warmed_up

    def test_tool_invoker_warm_up_with_mixed_toolsets(self):
        """Test that ToolInvoker.warm_up() works with combined toolsets using concatenation."""
        # Create first toolset with a tracking tool
        tool1 = WarmupTrackingTool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "tool1",
        )
        toolset1 = WarmupTrackingToolset([tool1])

        # Create second toolset with another tracking tool
        tool2 = WarmupTrackingTool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "tool2",
        )
        toolset2 = WarmupTrackingToolset([tool2])

        # Combine toolsets using the + operator (creates _ToolsetWrapper)
        combined = toolset1 + toolset2

        # Create invoker with the combined toolset
        invoker = ToolInvoker(tools=combined)

        assert not toolset1.was_warmed_up
        assert not toolset2.was_warmed_up

        invoker.warm_up()

        # Both toolsets should be warmed up
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up
