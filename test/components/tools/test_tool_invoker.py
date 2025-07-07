# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from haystack import Pipeline
from haystack.components.agents.state import State
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools.tool_invoker import StringConversionError, ToolInvoker, ToolNotFoundException
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


class TestToolInvoker:
    def test_init(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])

        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {"weather_tool": weather_tool}
        assert invoker.raise_on_failure
        assert not invoker.convert_result_to_json_string

    def test_init_with_toolset(self, tool_set):
        tool_invoker = ToolInvoker(tools=tool_set)
        assert tool_invoker.tools == tool_set
        assert tool_invoker._tools_with_names == {"weather_tool": tool_set.tools[0], "addition_tool": tool_set.tools[1]}

    def test_init_fails_wo_tools(self):
        with pytest.raises(ValueError):
            ToolInvoker(tools=[])

    def test_init_fails_with_duplicate_tool_names(self, weather_tool, faulty_tool):
        with pytest.raises(ValueError):
            ToolInvoker(tools=[weather_tool, weather_tool])

        new_tool = faulty_tool
        new_tool.name = "weather_tool"
        with pytest.raises(ValueError):
            ToolInvoker(tools=[weather_tool, new_tool])

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

    def test_run_with_streaming_callback(self, invoker):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[message], streaming_callback=streaming_callback)
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        # check we called the streaming callback
        assert streaming_callback_called

        tool_message = result["tool_messages"][0]
        assert isinstance(tool_message, ChatMessage)
        assert tool_message.is_from(ChatRole.TOOL)

        assert tool_message.tool_call_results
        tool_call_result = tool_message.tool_call_result

        assert isinstance(tool_call_result, ToolCallResult)
        assert tool_call_result.result == str({"weather": "mostly sunny", "temperature": 7, "unit": "celsius"})
        assert tool_call_result.origin == tool_call
        assert not tool_call_result.error

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
    async def test_run_async_with_streaming_callback(self, weather_tool):
        streaming_callback_called = False

        async def streaming_callback(chunk: StreamingChunk) -> None:
            print(f"Streaming callback called with chunk: {chunk}")
            nonlocal streaming_callback_called
            streaming_callback_called = True

        tool_invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=True, convert_result_to_json_string=False)

        tool_calls = [
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Paris"}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Rome"}),
        ]

        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        result = await tool_invoker.run_async(messages=[message], streaming_callback=streaming_callback)
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

        # check we called the streaming callback
        assert streaming_callback_called

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

    def test_run_with_toolset(self, tool_set):
        tool_invoker = ToolInvoker(tools=tool_set, raise_on_failure=True, convert_result_to_json_string=False)
        tool_call = ToolCall(tool_name="addition_tool", arguments={"num1": 5, "num2": 3})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = tool_invoker.run(messages=[message])
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        tool_message = result["tool_messages"][0]
        assert isinstance(tool_message, ChatMessage)
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_results

        tool_call_result = tool_message.tool_call_result
        assert isinstance(tool_call_result, ToolCallResult)
        assert tool_call_result.result == str(8)
        assert tool_call_result.origin == tool_call
        assert not tool_call_result.error

    @pytest.mark.asyncio
    async def test_run_async_with_toolset(self, tool_set):
        tool_invoker = ToolInvoker(tools=tool_set, raise_on_failure=True, convert_result_to_json_string=False)
        tool_calls = [
            ToolCall(tool_name="addition_tool", arguments={"num1": 5, "num2": 3}),
            ToolCall(tool_name="addition_tool", arguments={"num1": 5, "num2": 3}),
            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
        ]
        message = ChatMessage.from_assistant(tool_calls=tool_calls)

        result = await tool_invoker.run_async(messages=[message])
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
        assert not tool_call_result.error

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

    def test_parallel_tool_calling_with_state_updates(self):
        """Test that parallel tool execution with state updates works correctly with the state lock."""
        # Create a shared counter variable to simulate a state value that gets updated
        execution_log = []

        def function_1():
            time.sleep(0.1)
            execution_log.append("tool_1_executed")
            return {"counter": 1, "tool_name": "tool_1"}

        def function_2():
            time.sleep(0.1)
            execution_log.append("tool_2_executed")
            return {"counter": 2, "tool_name": "tool_2"}

        def function_3():
            time.sleep(0.1)
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
            time.sleep(0.1)
            execution_log.append("tool_1_executed")
            return {"counter": 1, "tool_name": "tool_1"}

        def function_2():
            time.sleep(0.1)
            execution_log.append("tool_2_executed")
            return {"counter": 2, "tool_name": "tool_2"}

        def function_3():
            time.sleep(0.1)
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


class TestMergeToolOutputs:
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
