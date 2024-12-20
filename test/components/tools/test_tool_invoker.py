import pytest
import datetime

from haystack import Pipeline

from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult, ChatRole
from haystack.dataclasses.tool import Tool, ToolInvocationError
from haystack.components.tools.tool_invoker import ToolInvoker, ToolNotFoundException, StringConversionError
from haystack.components.generators.chat.openai import OpenAIChatGenerator


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

    def test_run(self, invoker):
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[message])
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        tool_message = result["tool_messages"][0]
        assert isinstance(tool_message, ChatMessage)
        assert tool_message.is_from(ChatRole.TOOL)

        assert tool_message.tool_call_results
        tool_call_result = tool_message.tool_call_result

        assert isinstance(tool_call_result, ToolCallResult)
        assert tool_call_result.result == str({"weather": "mostly sunny", "temperature": 7, "unit": "celsius"})
        assert tool_call_result.origin == tool_call
        assert not tool_call_result.error

    def test_run_no_messages(self, invoker):
        result = invoker.run(messages=[])
        assert result == {"tool_messages": []}

    def test_run_no_tool_calls(self, invoker):
        user_message = ChatMessage.from_user(text="Hello!")
        assistant_message = ChatMessage.from_assistant(text="How can I help you?")

        result = invoker.run(messages=[user_message, assistant_message])
        assert result == {"tool_messages": []}

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

    def test_tool_not_found_error(self, invoker):
        tool_call = ToolCall(tool_name="non_existent_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        with pytest.raises(ToolNotFoundException):
            invoker.run(messages=[tool_call_message])

    def test_tool_not_found_does_not_raise_exception(self, invoker):
        invoker.raise_on_failure = False

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

    def test_tool_invocation_error_does_not_raise_exception(self, faulty_invoker):
        faulty_invoker.raise_on_failure = False

        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = faulty_invoker.run(messages=[tool_call_message])
        tool_message = result["tool_messages"][0]
        assert tool_message.tool_call_results[0].error
        assert "invocation failed" in tool_message.tool_call_results[0].result

    def test_string_conversion_error(self, invoker):
        invoker.convert_result_to_json_string = True

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        with pytest.raises(StringConversionError):
            invoker._prepare_tool_result_message(result=tool_result, tool_call=tool_call)

    def test_string_conversion_error_does_not_raise_exception(self, invoker):
        invoker.convert_result_to_json_string = True
        invoker.raise_on_failure = False

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        tool_message = invoker._prepare_tool_result_message(result=tool_result, tool_call=tool_call)

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
            },
        }

    def test_from_dict(self, weather_tool):
        data = {
            "type": "haystack.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
            },
        }
        invoker = ToolInvoker.from_dict(data)
        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {"weather_tool": weather_tool}
        assert invoker.raise_on_failure
        assert not invoker.convert_result_to_json_string

    def test_serde_in_pipeline(self, invoker, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        pipeline = Pipeline()
        pipeline.add_component("invoker", invoker)
        pipeline.add_component("chatgenerator", OpenAIChatGenerator())
        pipeline.connect("invoker", "chatgenerator")

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "invoker": {
                    "type": "haystack.components.tools.tool_invoker.ToolInvoker",
                    "init_parameters": {
                        "tools": [
                            {
                                "name": "weather_tool",
                                "description": "Provides weather information for a given location.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                    "required": ["location"],
                                },
                                "function": "tools.test_tool_invoker.weather_function",
                            }
                        ],
                        "raise_on_failure": True,
                        "convert_result_to_json_string": False,
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
                    },
                },
            },
            "connections": [{"sender": "invoker.tool_messages", "receiver": "chatgenerator.messages"}],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline
