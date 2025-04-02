import pytest
import datetime

from haystack import Pipeline


from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.tools.tool_invoker import ToolInvoker, ToolNotFoundException, StringConversionError
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult, ChatRole
from haystack.dataclasses.state import State
from haystack.tools import ComponentTool, Tool
from haystack.tools.errors import ToolInvocationError
from haystack.tools.toolset import Toolset


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


# Define a custom Toolset subclass
class CustomToolset(Toolset):
    def __init__(self, tools, custom_attr):
        super().__init__(tools)
        self.custom_attr = custom_attr

    def to_dict(self):
        data = super().to_dict()
        data["custom_attr"] = self.custom_attr
        return data

    @classmethod
    def from_dict(cls, data):
        tools = [Tool.from_dict(tool_data) for tool_data in data["data"]["tools"]]
        custom_attr = data["custom_attr"]
        return cls(tools=tools, custom_attr=custom_attr)


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

    def test_inject_state_args_no_tool_inputs(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = ToolInvoker._inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_no_tool_inputs_component_tool(self):
        comp = PromptBuilder(template="Hello, {{name}}!")
        prompt_tool = ComponentTool(
            component=comp, name="prompt_tool", description="Creates a personalized greeting prompt."
        )
        state = State(schema={"name": {"type": str}}, data={"name": "James"})
        args = ToolInvoker._inject_state_args(tool=prompt_tool, llm_args={}, state=state)
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
        args = ToolInvoker._inject_state_args(tool=weather_tool, llm_args={}, state=state)
        assert args == {"location": "Berlin"}

    def test_inject_state_args_param_in_state_and_llm(self):
        weather_tool = Tool(
            name="weather_tool",
            description="Provides weather information for a given location.",
            parameters=weather_parameters,
            function=weather_function,
        )
        state = State(schema={"location": {"type": str}}, data={"location": "Berlin"})
        args = ToolInvoker._inject_state_args(tool=weather_tool, llm_args={"location": "Paris"}, state=state)
        assert args == {"location": "Paris"}

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

    def test_string_conversion_error(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=True, convert_result_to_json_string=True)

        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})

        tool_result = datetime.datetime.now()
        with pytest.raises(StringConversionError):
            invoker._prepare_tool_result_message(result=tool_result, tool_call=tool_call, tool_to_invoke=weather_tool)

    def test_string_conversion_error_does_not_raise_exception(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool], raise_on_failure=False, convert_result_to_json_string=True)

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
        pipeline.connect("invoker.tool_messages", "chatgenerator.messages")

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

    def test_init_with_toolset(self, weather_tool):
        # Create a Toolset with the weather_tool
        toolset = Toolset(tools=[weather_tool])

        # Initialize ToolInvoker with the Toolset
        invoker = ToolInvoker(tools=toolset)

        # Assert that the ToolInvoker contains the tools from the Toolset
        assert invoker.tools == toolset
        assert invoker._tools_with_names == {tool.name: tool for tool in toolset}

    def test_run_with_toolset(self, weather_tool):
        # Create a Toolset with the weather_tool
        toolset = Toolset(tools=[weather_tool])

        # Initialize ToolInvoker with the Toolset
        invoker = ToolInvoker(tools=toolset)

        # Create a ToolCall for the weather_tool
        tool_call = ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Run the ToolInvoker with the message
        result = invoker.run(messages=[message])

        # Assert that the tool invocation is successful
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

    def test_serde_with_toolset(self, weather_tool):
        # Create a Toolset with the weather_tool
        toolset = Toolset(tools=[weather_tool])

        # Initialize ToolInvoker with the Toolset
        invoker = ToolInvoker(tools=toolset)

        # Serialize the ToolInvoker to a dictionary
        data = invoker.to_dict()

        # Deserialize the dictionary back to a ToolInvoker
        deserialized_invoker = ToolInvoker.from_dict(data)

        # Assert that the deserialized ToolInvoker is equivalent to the original
        assert deserialized_invoker.tools == invoker.tools
        assert deserialized_invoker._tools_with_names == invoker._tools_with_names
        assert deserialized_invoker.raise_on_failure == invoker.raise_on_failure
        assert deserialized_invoker.convert_result_to_json_string == invoker.convert_result_to_json_string

    def test_tool_invocation_error_with_toolset(self, faulty_tool):
        # Create a Toolset with the faulty_tool
        toolset = Toolset(tools=[faulty_tool])

        # Initialize ToolInvoker with the Toolset
        invoker = ToolInvoker(tools=toolset)

        # Create a ToolCall for the faulty_tool
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Assert that the tool invocation raises a ToolInvocationError
        with pytest.raises(ToolInvocationError):
            invoker.run(messages=[tool_call_message])

    def test_toolinvoker_deserialization_with_custom_toolset(self, weather_tool):
        # Initialize the custom Toolset
        custom_toolset = CustomToolset(tools=[weather_tool], custom_attr="custom_value")

        # Initialize ToolInvoker with the custom Toolset
        invoker = ToolInvoker(tools=custom_toolset)

        # Serialize the ToolInvoker to a dictionary
        data = invoker.to_dict()

        # Assert the structure and content of the serialized data
        assert isinstance(data, dict), "Serialized data should be a dictionary"
        assert "type" in data and "init_parameters" in data, (
            "Serialized data should contain 'type' and 'init_parameters' keys"
        )

        # Check the tools data
        tools_data = data["init_parameters"]["tools"]
        assert isinstance(tools_data, dict), "Tools data should be a dictionary because it's a Toolset"
        assert len(tools_data["data"]["tools"]) == 1, "There should be one tool in the serialized data"
        assert tools_data["data"]["tools"][0]["type"] == "haystack.tools.tool.Tool", (
            "Tool type should be 'haystack.tools.tool.Tool'"
        )

        # Check custom attributes within the Toolset
        assert tools_data.get("custom_attr") == "custom_value", "Custom attribute should be 'custom_value'"

        # Deserialize the dictionary back to a ToolInvoker
        deserialized_invoker = ToolInvoker.from_dict(data)

        # Assert that the deserialized ToolInvoker is equivalent to the original
        assert deserialized_invoker.tools == invoker.tools
        assert deserialized_invoker._tools_with_names == invoker._tools_with_names
        assert deserialized_invoker.raise_on_failure == invoker.raise_on_failure
        assert deserialized_invoker.convert_result_to_json_string == invoker.convert_result_to_json_string

    def test_openai_chat_generator_with_toolset(self, weather_tool):
        # Create a Toolset with the weather tool
        toolset = Toolset([weather_tool])

        # Initialize OpenAIChatGenerator with the Toolset
        generator = OpenAIChatGenerator(tools=toolset)

        # Serialize the generator to a dictionary
        data = generator.to_dict()

        # Assert the structure and content of the serialized data
        assert isinstance(data, dict), "Serialized data should be a dictionary"
        assert "tools" in data["init_parameters"], "Serialized data should contain 'tools' key in 'init_parameters'"

        # Check the tools data
        tools_data = data["init_parameters"]["tools"]
        assert isinstance(tools_data, dict), "Tools data should be a dictionary"
        assert len(tools_data["data"]["tools"]) == 1, "There should be one tool in the serialized data"
        assert isinstance(tools_data["data"]["tools"][0], dict), "Each tool should be serialized as a dictionary"
        assert tools_data["data"]["tools"][0]["type"] == "haystack.tools.tool.Tool", (
            "Tool type should be 'haystack.tools.tool.Tool'"
        )

        # Deserialize the dictionary back to an OpenAIChatGenerator
        deserialized_generator = OpenAIChatGenerator.from_dict(data)

        # Assert that the deserialized generator has the same tools
        assert len(deserialized_generator.tools) == len(generator.tools), (
            "Deserialized generator should have the same number of tools"
        )


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
