# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.components.agents.state import State
from haystack.components.agents.tool_invoker import (
    ToolNotFoundException,
    _build_tool_result_message,
    _inject_state_args,
    _merge_tool_outputs_into_state,
    _process_tool_output,
    _result_to_string,
    _validate_and_prepare_tools,
    run_tool,
    run_tool_async,
)
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ImageContent,
    StreamingChunk,
    TextContent,
    ToolCall,
    ToolCallResult,
)
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


class TestCore:
    def test_validate_and_prepare_tools(self, weather_tool, faulty_tool):
        result = _validate_and_prepare_tools([weather_tool, faulty_tool])
        assert result == {"weather_tool": weather_tool, "faulty_tool": faulty_tool}

        toolset = Toolset([weather_tool, faulty_tool])
        result = _validate_and_prepare_tools(toolset)
        assert result == {"weather_tool": weather_tool, "faulty_tool": faulty_tool}

    def test_validate_and_prepare_tools_validation_failures(self, weather_tool):
        with pytest.raises(ValueError):
            _validate_and_prepare_tools([])

        with pytest.raises(ValueError):
            _validate_and_prepare_tools([weather_tool, weather_tool])

    def test_inject_state_args_no_tool_inputs(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = _inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_no_tool_inputs_component_tool(self):
        comp = PromptBuilder(template="Hello, {{name}}!")
        prompt_tool = ComponentTool(
            component=comp, name="prompt_tool", description="Creates a personalized greeting prompt."
        )
        state = State(schema={"name": {"type": str}}, data={"name": "James"})
        args = _inject_state_args(tool=prompt_tool, llm_args={}, state=state)
        assert args == {"name": "James"}

    def test_inject_state_args_with_tool_inputs(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            inputs_from_state={"loc": "location"},
        )
        state = State(schema={"location": {"type": str}}, data={"loc": "Berlin"})
        args = _inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_param_in_state_and_llm(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = _inject_state_args(tool=weather_tool, llm_args={"location": "Paris"}, state=state)
        assert args == {"location": "Paris"}

    def test_inject_state_args_injects_state_object_for_state_annotated_param(self):
        def function_with_state(city: str, state: State) -> str:
            return f"Weather in {city}"

        state_tool = Tool(
            name="state_tool",
            description="A tool that receives the live State object.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=function_with_state,
        )
        state = State(schema={"city": {"type": str}}, data={"city": "Berlin"})
        args = _inject_state_args(tool=state_tool, llm_args={"city": "Paris"}, state=state)
        assert args["city"] == "Paris"
        assert args["state"] is state

    def test_inject_state_args_injects_state_object_for_optional_state_annotated_param(self):
        def function_with_optional_state(city: str, state: State | None = None) -> str:
            return f"Weather in {city}"

        state_tool = Tool(
            name="state_tool",
            description="A tool that receives an optional State object.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=function_with_optional_state,
        )
        state = State(schema={})
        args = _inject_state_args(tool=state_tool, llm_args={"city": "Paris"}, state=state)
        assert args["city"] == "Paris"
        assert args["state"] is state


class TestRunTool:
    def test_run_with_streaming_callback_finish_reason(self, weather_tool):
        streaming_chunks = []

        def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = run_tool(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages) == 1
        assert len(streaming_chunks) >= 2
        final_chunk = streaming_chunks[-1]
        assert final_chunk.finish_reason == "tool_call_results"
        assert final_chunk.meta["finish_reason"] == "tool_call_results"
        assert final_chunk.content == ""

    @pytest.mark.asyncio
    async def test_run_async_with_streaming_callback_finish_reason(self, weather_tool):
        streaming_chunks = []

        async def streaming_callback(chunk: StreamingChunk) -> None:
            streaming_chunks.append(chunk)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = await run_tool_async(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages) == 1
        assert len(streaming_chunks) >= 2
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
        with patch("haystack.components.generators.chat.OpenAIChatGenerator.run") as mock_run:
            mock_run.return_value = {"replies": [ChatMessage.from_assistant("Hello! How can I help you?")]}
            run_tool(
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
                state=State(schema={}),
                tools=[llm_tool],
                streaming_callback=print_streaming_chunk,
                enable_streaming_callback_passthrough=True,
            )
            mock_run.assert_called_once_with(
                messages=[ChatMessage.from_user(text="Hello!")], streaming_callback=print_streaming_chunk
            )

    def test_run_no_messages(self, weather_tool):
        tool_messages, state = run_tool(messages=[], state=State(schema={}), tools=[weather_tool])
        assert tool_messages == []

    def test_run_no_tool_calls(self, weather_tool):
        user_message = ChatMessage.from_user(text="Hello!")
        assistant_message = ChatMessage.from_assistant(text="How can I help you?")

        tool_messages, _ = run_tool(
            messages=[user_message, assistant_message], state=State(schema={}), tools=[weather_tool]
        )
        assert tool_messages == []

    def test_run_multiple_tool_calls(self, weather_tool):
        tool_calls = [
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Rome"}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        tool_messages, _ = run_tool(messages=[message], state=State(schema={}), tools=[weather_tool])
        assert len(tool_messages) == 3

        for i, tool_message in enumerate(tool_messages):
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
        tool_call = ToolCall(tool_name="hello_world", arguments={})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = run_tool(messages=[tool_call_message], state=State(schema={}), tools=[hello_world_tool])
        assert len(tool_messages) == 1

        tool_call_result = tool_messages[0].tool_call_result
        assert isinstance(tool_call_result, ToolCallResult)
        assert not tool_call_result.error
        assert tool_call_result.result == "Hello, world!"

    def test_run_with_tools(self, weather_tool):
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = run_tool(messages=[message], state=State(schema={}), tools=[weather_tool])

        tool_call_result = tool_messages[0].tool_call_result
        assert not tool_call_result.error
        assert tool_call_result.result == '{"weather": "mostly sunny", "temperature": 7, "unit": "celsius"}'
        assert tool_call_result.origin == tool_call

    @pytest.mark.asyncio
    async def test_run_async_with_tools(self, weather_tool):
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = await run_tool_async(messages=[message], state=State(schema={}), tools=[weather_tool])
        tool_call_result = tool_messages[0].tool_call_result
        assert not tool_call_result.error
        assert tool_call_result.result == '{"weather": "mostly sunny", "temperature": 7, "unit": "celsius"}'
        assert tool_call_result.origin == tool_call

    def test_parallel_tool_calling_with_state_updates(self):
        """Test that parallel tool execution with state updates works correctly with the state lock."""
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

        state = State(schema={"counter": {"type": int}, "last_tool": {"type": str}})
        tool_calls = [
            ToolCall(tool_name="state_tool_1", arguments={}),
            ToolCall(tool_name="state_tool_2", arguments={}),
            ToolCall(tool_name="state_tool_3", arguments={}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)
        run_tool(messages=[message], state=state, tools=[tool_1, tool_2, tool_3])

        assert len(execution_log) == 3
        assert "tool_1_executed" in execution_log
        assert "tool_2_executed" in execution_log
        assert "tool_3_executed" in execution_log
        assert state.has("counter")
        assert state.has("last_tool")
        assert state.get("counter") in [1, 2, 3]
        assert state.get("last_tool") in ["tool_1", "tool_2", "tool_3"]

    @pytest.mark.asyncio
    async def test_async_parallel_tool_calling_with_state_updates(self):
        """Test that parallel tool execution with state updates works correctly with the state lock."""
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

        state = State(schema={"counter": {"type": int}, "last_tool": {"type": str}})
        tool_calls = [
            ToolCall(tool_name="state_tool_1", arguments={}),
            ToolCall(tool_name="state_tool_2", arguments={}),
            ToolCall(tool_name="state_tool_3", arguments={}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)
        await run_tool_async(messages=[message], state=state, tools=[tool_1, tool_2, tool_3])

        assert len(execution_log) == 3
        assert "tool_1_executed" in execution_log
        assert "tool_2_executed" in execution_log
        assert "tool_3_executed" in execution_log
        assert state.has("counter")
        assert state.has("last_tool")
        assert state.get("counter") in [1, 2, 3]
        assert state.get("last_tool") in ["tool_1", "tool_2", "tool_3"]

    def test_two_subsequent_run_tool_calls(self, weather_tool):
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

        tool_messages_1, _ = run_tool(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages_1) == 3

        tool_messages_2, _ = run_tool(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages_2) == 3

    @pytest.mark.asyncio
    async def test_two_subsequent_run_tool_async_calls(self, weather_tool):
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

        tool_messages_1, _ = await run_tool_async(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages_1) == 3

        tool_messages_2, _ = await run_tool_async(
            messages=[message], state=State(schema={}), tools=[weather_tool], streaming_callback=streaming_callback
        )
        assert len(tool_messages_2) == 3

    def test_run_injects_state_object_into_tool(self):
        received_state = {}

        def function_with_state(city: str, state: State) -> str:
            received_state["state"] = state
            return f"Weather in {city}: sunny"

        state_tool = Tool(
            name="state_tool",
            description="A tool that receives the live State object.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=function_with_state,
        )
        state = State(schema={"city": {"type": str}})

        tool_call = ToolCall(tool_name="state_tool", arguments={"city": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])
        tool_messages, _ = run_tool(messages=[message], state=state, tools=[state_tool])

        assert len(tool_messages) == 1
        assert not tool_messages[0].tool_call_results[0].error
        assert received_state["state"] is state

    @pytest.mark.asyncio
    async def test_run_async_injects_state_object_into_tool(self):
        received_state = {}

        def function_with_state(city: str, state: State) -> str:
            received_state["state"] = state
            return f"Weather in {city}: sunny"

        state_tool = Tool(
            name="state_tool",
            description="A tool that receives the live State object.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=function_with_state,
        )
        state = State(schema={"city": {"type": str}})

        tool_call = ToolCall(tool_name="state_tool", arguments={"city": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])
        tool_messages, _ = await run_tool_async(messages=[message], state=state, tools=[state_tool])

        assert len(tool_messages) == 1
        assert not tool_messages[0].tool_call_results[0].error
        assert received_state["state"] is state


class TestRunToolErrorHandling:
    def test_tool_not_found_error(self, weather_tool):
        tool_call = ToolCall(tool_name="non_existent_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        with pytest.raises(ToolNotFoundException):
            run_tool(messages=[tool_call_message], state=State(schema={}), tools=[weather_tool])

    def test_tool_not_found_does_not_raise_exception(self, weather_tool):
        tool_call = ToolCall(tool_name="non_existent_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = run_tool(
            messages=[tool_call_message], state=State(schema={}), tools=[weather_tool], raise_on_failure=False
        )
        tool_message = tool_messages[0]
        assert tool_message.tool_call_results[0].error
        assert "not found" in tool_message.tool_call_results[0].result

    def test_tool_invocation_error(self, faulty_tool):
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        with pytest.raises(ToolInvocationError):
            run_tool(messages=[tool_call_message], state=State(schema={}), tools=[faulty_tool])

    def test_tool_invocation_error_does_not_raise_exception(self, faulty_tool):
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        tool_messages, _ = run_tool(
            messages=[tool_call_message], state=State(schema={}), tools=[faulty_tool], raise_on_failure=False
        )
        tool_message = tool_messages[0]
        assert tool_message.tool_call_results[0].error
        assert "Failed to invoke" in tool_message.tool_call_results[0].result

    def test_outputs_to_string_with_multiple_outputs(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_string={"weather": {"source": "weather"}, "temp": {"source": "temperature"}},
        )
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = {"weather": "sunny", "temperature": 25, "unit": "celsius"}
        chat_message = _build_tool_result_message(tool_result, tool_call, weather_tool)
        assert chat_message.tool_call_results[0].result == '{"weather": "sunny", "temp": "25"}'

    def test_output_handler_failure_falls_back_to_string(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_string={"handler": json.dumps},
        )
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        tool_message = _build_tool_result_message(tool_result, tool_call, weather_tool)

        assert not tool_message.tool_call_results[0].error
        assert isinstance(tool_message.tool_call_results[0].result, str)

    def test_output_handler_failure_falls_back_to_string_raw_result(self):
        def handler(result):
            raise ValueError("Handler error")

        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_string={"handler": handler, "raw_result": True},
        )
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_message = _build_tool_result_message("something", tool_call, weather_tool)
        assert not tool_message.tool_call_results[0].error
        assert tool_message.tool_call_results[0].result == "something"

    def test_run_state_merge_error_always_raises(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        with pytest.raises(RuntimeError, match="weather_tool"):
            run_tool(
                messages=[message], state=state, tools=[weather_tool_with_outputs_to_state], raise_on_failure=False
            )

    @pytest.mark.asyncio
    async def test_run_async_state_merge_error_always_raises(self, weather_tool_with_outputs_to_state):
        class ProblematicState(State):
            def set(self, key: str, value: Any, handler_override=None):
                raise ValueError("State set operation failed")

        state = ProblematicState(schema={"test_key": {"type": str}})
        tool_calls = [ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        with pytest.raises(RuntimeError, match="weather_tool"):
            await run_tool_async(
                messages=[message], state=state, tools=[weather_tool_with_outputs_to_state], raise_on_failure=False
            )


class TestUtilities:
    def test_result_to_string(self):
        assert _result_to_string("hello") == "hello"
        assert _result_to_string(42) == "42"
        assert _result_to_string(3.14) == "3.14"
        assert _result_to_string(True) == "true"
        assert _result_to_string(None) == "null"

        assert _result_to_string([1, 2, 3]) == "[1, 2, 3]"
        assert _result_to_string({"key": "value"}) == '{"key": "value"}'

        assert _result_to_string("Hello 🌍") == "Hello 🌍"

    def test_result_to_string_with_serializable_objects(self):
        class MockObject:
            def __init__(self, value):
                self.value = value

            def to_dict(self):
                return {"value": self.value}

        mock_obj = MockObject("test_value")
        result = _result_to_string(mock_obj)

        assert "test_value" in result
        assert "value" in result

    def test_merge_tool_outputs_result_not_a_dict(self, weather_tool):
        state = State(schema={"weather": {"type": str}})
        _merge_tool_outputs_into_state(tool=weather_tool, result="test", state=state)
        assert state.data == {}

    def test_merge_tool_outputs_empty_dict(self, weather_tool):
        state = State(schema={"weather": {"type": str}})
        _merge_tool_outputs_into_state(tool=weather_tool, result={}, state=state)
        assert state.data == {}

    def test_merge_tool_outputs_no_output_mapping(self, weather_tool):
        state = State(schema={"weather": {"type": str}})
        _merge_tool_outputs_into_state(
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
        state = State(schema={"weather": {"type": str}})
        _merge_tool_outputs_into_state(
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
        state = State(schema={"all_weather_results": {"type": str}})
        _merge_tool_outputs_into_state(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {"all_weather_results": {"weather": "sunny", "temperature": 14, "unit": "celsius"}}

    def test_merge_tool_outputs_source_key_absent_does_not_corrupt_list_state(self):
        """
        Simulates a PipelineTool wrapping a pipeline with a conditional branch that may not execute, resulting in the
        source key being absent from the tool result. The test verifies that in this case, the existing list in state
        is not corrupted by appending None.
        """
        tool = Tool(
            name="retrieval",
            description="mock",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            function=lambda query: {},
            outputs_to_state={"documents": {"source": "documents_output"}},
        )
        existing_doc = Document(content="from first call")
        state = State(schema={"documents": {"type": list[Document]}})
        state.set("documents", [existing_doc])

        _merge_tool_outputs_into_state(tool=tool, result={"result": "no web results found"}, state=state)

        assert state.data["documents"] == [existing_doc]
        assert None not in state.data["documents"]

    def test_merge_tool_outputs_with_output_mapping_and_handler(self):
        handler = lambda _, new: f"{new}"  # noqa: E731
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
            outputs_to_state={"temperature": {"source": "temperature", "handler": handler}},
        )
        state = State(schema={"temperature": {"type": str}})
        _merge_tool_outputs_into_state(
            tool=weather_tool, result={"weather": "sunny", "temperature": 14, "unit": "celsius"}, state=state
        )
        assert state.data == {"temperature": "14"}

    def test_process_output_empty_config(self, base64_image_string):
        image_content = ImageContent(base64_image=base64_image_string, mime_type="image/png")

        result = _process_tool_output(
            config={"raw_result": True},
            result=[image_content],
            tool_call=ToolCall(tool_name="retrieve_image", arguments={}),
        )
        assert result == [image_content]

    def test_process_output_source_only(self, base64_image_string):
        image_content = ImageContent(base64_image=base64_image_string, mime_type="image/png")

        result = _process_tool_output(
            config={"source": "images", "raw_result": True},
            result={"images": [image_content]},
            tool_call=ToolCall(tool_name="retrieve_image", arguments={}),
        )
        assert result == [image_content]

    def test_process_output_handler_only(self, base64_image_string):
        def handler(result: dict) -> list[ImageContent]:
            return [ImageContent(base64_image=result["base64_image_string"], mime_type=result["mime_type"])]

        result = _process_tool_output(
            config={"handler": handler, "raw_result": True},
            result={"base64_image_string": base64_image_string, "mime_type": "image/png"},
            tool_call=ToolCall(tool_name="retrieve_image", arguments={}),
        )
        assert result == [ImageContent(base64_image=base64_image_string, mime_type="image/png")]

    def test_process_output_source_and_handler(self, base64_image_string):
        def handler(result: dict) -> list[ImageContent]:
            return [ImageContent(base64_image=result["base64_image_string"], mime_type=result["mime_type"])]

        result = _process_tool_output(
            config={"source": "images", "handler": handler, "raw_result": True},
            result={
                "images": {"base64_image_string": base64_image_string, "mime_type": "image/png"},
                "other_key": "other_value",
            },
            tool_call=ToolCall(tool_name="retrieve_image", arguments={}),
        )
        assert result == [ImageContent(base64_image=base64_image_string, mime_type="image/png")]

    def test_output_to_result_e2e(self, weather_tool):
        def handler(result):
            return [
                TextContent(text=f"weather: {result['weather']}"),
                TextContent(text=f"temperature: {result['temperature']} {result['unit']}"),
            ]

        weather_tool.outputs_to_string = {"handler": handler, "raw_result": True}

        message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        )

        tool_messages, _ = run_tool(messages=[message], state=State(schema={}), tools=[weather_tool])

        assert tool_messages[0].tool_call_results[0].result == [
            TextContent(text="weather: mostly sunny"),
            TextContent(text="temperature: 7 celsius"),
        ]
