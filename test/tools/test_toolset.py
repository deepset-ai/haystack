# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.converters import OutputAdapter
from haystack.dataclasses.chat_message import ChatRole, ToolCall, ToolCallResult
from haystack.tools import Tool, Toolset
from haystack.tools.errors import ToolInvocationError


# Common functions for tests
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def subtract_numbers(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


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


def test_toolset_with_multiple_tools():
    """Test that a Toolset with multiple tools works properly."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )

    toolset = Toolset([add_tool, multiply_tool])

    assert len(toolset) == 2
    assert toolset[0].name == "add"
    assert toolset[1].name == "multiply"

    invoker = ToolInvoker(tools=toolset)

    add_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    add_message = ChatMessage.from_assistant(tool_calls=[add_call])

    multiply_call = ToolCall(tool_name="multiply", arguments={"a": 4, "b": 5})
    multiply_message = ChatMessage.from_assistant(tool_calls=[multiply_call])

    result = invoker.run(messages=[add_message, multiply_message])

    assert len(result["tool_messages"]) == 2
    tool_results = [message.tool_call_result.result for message in result["tool_messages"]]
    assert "5" in tool_results
    assert "20" in tool_results


def test_toolset_adding():
    """Test that tools can be added to a Toolset."""
    toolset = Toolset()
    assert len(toolset) == 0

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    toolset.add(add_tool)
    assert len(toolset) == 1
    assert toolset[0].name == "add"

    invoker = ToolInvoker(tools=toolset)
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])
    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_toolset_addition():
    """Test that toolsets can be combined."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )

    subtract_tool = Tool(
        name="subtract",
        description="Subtract two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=subtract_numbers,
    )

    toolset1 = Toolset([add_tool])
    toolset2 = Toolset([multiply_tool])

    combined_toolset = toolset1 + toolset2
    assert len(combined_toolset) == 2

    combined_toolset = combined_toolset + subtract_tool
    assert len(combined_toolset) == 3

    tool_names = [tool.name for tool in combined_toolset]
    assert "add" in tool_names
    assert "multiply" in tool_names
    assert "subtract" in tool_names

    invoker = ToolInvoker(tools=combined_toolset)

    add_call = ToolCall(tool_name="add", arguments={"a": 10, "b": 5})
    multiply_call = ToolCall(tool_name="multiply", arguments={"a": 10, "b": 5})
    subtract_call = ToolCall(tool_name="subtract", arguments={"a": 10, "b": 5})

    message = ChatMessage.from_assistant(tool_calls=[add_call, multiply_call, subtract_call])

    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 3
    tool_results = [message.tool_call_result.result for message in result["tool_messages"]]
    assert "15" in tool_results
    assert "50" in tool_results
    assert "5" in tool_results


def test_toolset_contains():
    """Test that the __contains__ method works correctly."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )

    toolset = Toolset([add_tool])

    # Test with a tool instance
    assert add_tool in toolset
    assert multiply_tool not in toolset

    # Test with a tool name
    assert "add" in toolset
    assert "multiply" not in toolset
    assert "non_existent_tool" not in toolset


def test_toolset_add_various_types():
    """Test that the __add__ method works with various object types."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )

    subtract_tool = Tool(
        name="subtract",
        description="Subtract two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=subtract_numbers,
    )

    # Test adding a single tool
    toolset1 = Toolset([add_tool])
    result1 = toolset1 + multiply_tool
    assert len(result1) == 2
    assert add_tool in result1
    assert multiply_tool in result1

    # Test adding another toolset
    toolset2 = Toolset([subtract_tool])
    result2 = toolset1 + toolset2
    assert len(result2) == 2
    assert add_tool in result2
    assert subtract_tool in result2

    # Test adding a list of tools
    result3 = toolset1 + [multiply_tool, subtract_tool]
    assert len(result3) == 3
    assert add_tool in result3
    assert multiply_tool in result3
    assert subtract_tool in result3

    # Test adding types that aren't supported
    with pytest.raises(TypeError):
        toolset1 + "not_a_tool"

    with pytest.raises(TypeError):
        toolset1 + 123


def test_toolset_serialization():
    """Test that a Toolset can be serialized and deserialized."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    toolset = Toolset([add_tool])

    serialized = toolset.to_dict()

    deserialized = Toolset.from_dict(serialized)

    assert len(deserialized) == 1
    assert deserialized[0].name == "add"
    assert deserialized[0].description == "Add two numbers"

    invoker = ToolInvoker(tools=deserialized)
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])
    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_custom_toolset_serialization():
    """Test serialization and deserialization of a custom Toolset subclass."""
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    custom_attr_value = "custom_value"
    custom_toolset = CustomToolset(tools=[add_tool], custom_attr=custom_attr_value)

    serialized = custom_toolset.to_dict()
    assert serialized["type"].endswith("CustomToolset")
    assert serialized["custom_attr"] == custom_attr_value
    assert len(serialized["data"]["tools"]) == 1
    assert serialized["data"]["tools"][0]["data"]["name"] == "add"

    deserialized = CustomToolset.from_dict(serialized)
    assert isinstance(deserialized, CustomToolset)
    assert deserialized.custom_attr == custom_attr_value
    assert len(deserialized) == 1
    assert deserialized[0].name == "add"

    invoker = ToolInvoker(tools=deserialized)
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])
    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_toolset_duplicate_tool_names():
    """Test that a Toolset raises an error for duplicate tool names."""
    add_tool1 = Tool(
        name="add",
        description="Add two numbers (first)",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    add_tool2 = Tool(
        name="add",
        description="Add two numbers (second)",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )

    with pytest.raises(ValueError, match="Duplicate tool names found"):
        Toolset([add_tool1, add_tool2])

    toolset = Toolset([add_tool1])

    with pytest.raises(ValueError, match="Duplicate tool names found"):
        toolset.add(add_tool2)

    toolset2 = Toolset([add_tool2])
    with pytest.raises(ValueError, match="Duplicate tool names found"):
        combined = toolset + toolset2


@pytest.mark.integration
class TestToolsetIntegration:
    """Integration tests for Toolset in complete pipelines."""

    def test_basic_math_pipeline(self):
        """Test basic math operations through a complete pipeline flow."""
        add_tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers,
        )

        subtract_tool = Tool(
            name="subtract",
            description="Subtract two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=subtract_numbers,
        )

        math_toolset = Toolset([add_tool, subtract_tool])

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=math_toolset))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=math_toolset))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is 5 plus 3?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        pipe_result = result["response_llm"]["replies"][0].text
        assert "8" in pipe_result or "eight" in pipe_result

    def test_combined_toolsets_pipeline(self):
        """Test combining multiple toolsets in a complete pipeline."""
        add_tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers,
        )

        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=multiply_numbers,
        )

        basic_math_toolset = Toolset([add_tool])
        advanced_math_toolset = Toolset([multiply_tool])

        combined_toolset = basic_math_toolset + advanced_math_toolset

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=combined_toolset))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=combined_toolset))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is 6 times 7?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        pipe_result = result["response_llm"]["replies"][0].text
        assert "42" in pipe_result or "forty two" in pipe_result

    def test_custom_calculator_pipeline(self):
        """Test a custom calculator toolset in a complete pipeline."""

        class CalculatorToolset(Toolset):
            """A toolset for calculator operations."""

            def __init__(self):
                super().__init__([])
                self._create_tools()

            def _create_tools(self):
                add_tool = Tool(
                    name="add",
                    description="Add two numbers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                    function=add_numbers,
                )

                multiply_tool = Tool(
                    name="multiply",
                    description="Multiply two numbers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                    function=multiply_numbers,
                )

                self.add(add_tool)
                self.add(multiply_tool)

        calculator_toolset = CalculatorToolset()

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=calculator_toolset))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=calculator_toolset))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is 15 plus 7?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        pipe_result = result["response_llm"]["replies"][0].text
        assert "22" in pipe_result or "twenty two" in pipe_result

    def test_serde_in_pipeline(self, monkeypatch):
        """Test serialization and deserialization of a pipeline with toolsets."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        add_tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers,
        )

        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=multiply_numbers,
        )

        basic_toolset = Toolset([add_tool])
        advanced_toolset = Toolset([multiply_tool])
        combined_toolset = basic_toolset + advanced_toolset

        pipeline = Pipeline()
        pipeline.add_component("tool_invoker", ToolInvoker(tools=combined_toolset))
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo", tools=combined_toolset))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        pipeline_dict = pipeline.to_dict()
        assert (
            pipeline_dict["components"]["tool_invoker"]["type"] == "haystack.components.tools.tool_invoker.ToolInvoker"
        )
        assert len(pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]) == 2

        tools_dict = pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]
        tool_names = [tool["data"]["name"] for tool in tools_dict]
        assert "add" in tool_names
        assert "multiply" in tool_names

        pipeline_dict = pipeline.to_dict()
        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline == pipeline

    def test_toolset_serde_in_pipeline(self):
        """Test serialization and deserialization of toolsets within a pipeline."""

        class CalculatorToolset(Toolset):
            """A toolset for calculator operations."""

            def __init__(self):
                super().__init__([])
                self._create_tools()

            def _create_tools(self):
                add_tool = Tool(
                    name="add",
                    description="Add two numbers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                    function=add_numbers,
                )

                self.add(add_tool)

        calculator_toolset = CalculatorToolset()

        pipeline = Pipeline()
        pipeline.add_component("tool_invoker", ToolInvoker(tools=calculator_toolset))

        pipeline_dict = pipeline.to_dict()

        tool_invoker_dict = pipeline_dict["components"]["tool_invoker"]
        assert tool_invoker_dict["type"] == "haystack.components.tools.tool_invoker.ToolInvoker"
        assert len(tool_invoker_dict["init_parameters"]["tools"]) == 1

        tool_dict = tool_invoker_dict["init_parameters"]["tools"][0]
        assert tool_dict["data"]["name"] == "add"
        assert tool_dict["data"]["description"] == "Add two numbers"

        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline


class TestToolsetWithToolInvoker:
    def test_toolset_with_tool_invoker(self):
        """Test that a Toolset works with ToolInvoker."""
        add_tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers,
        )

        toolset = Toolset([add_tool])

        invoker = ToolInvoker(tools=toolset)

        tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = invoker.run(messages=[message])

        assert len(result["tool_messages"]) == 1
        assert result["tool_messages"][0].tool_call_result.result == "5"

    def test_init_with_toolset(self, weather_tool):
        """Test initializing ToolInvoker with a Toolset."""
        toolset = Toolset(tools=[weather_tool])
        invoker = ToolInvoker(tools=toolset)
        assert invoker.tools == toolset
        assert invoker._tools_with_names == {tool.name: tool for tool in toolset}

    def test_run_with_toolset(self, weather_tool):
        """Test running ToolInvoker with a Toolset."""
        toolset = Toolset(tools=[weather_tool])
        invoker = ToolInvoker(tools=toolset)
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

    def test_serde_with_toolset(self, weather_tool):
        """Test serialization and deserialization of ToolInvoker with a Toolset."""
        toolset = Toolset(tools=[weather_tool])
        invoker = ToolInvoker(tools=toolset)
        data = invoker.to_dict()
        deserialized_invoker = ToolInvoker.from_dict(data)
        assert deserialized_invoker.tools == invoker.tools
        assert deserialized_invoker._tools_with_names == invoker._tools_with_names
        assert deserialized_invoker.raise_on_failure == invoker.raise_on_failure
        assert deserialized_invoker.convert_result_to_json_string == invoker.convert_result_to_json_string

    def test_tool_invocation_error_with_toolset(self, faulty_tool):
        """Test tool invocation errors with a Toolset."""
        toolset = Toolset(tools=[faulty_tool])
        invoker = ToolInvoker(tools=toolset)
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])
        with pytest.raises(ToolInvocationError):
            invoker.run(messages=[tool_call_message])

    def test_toolinvoker_deserialization_with_custom_toolset(self, weather_tool):
        """Test deserialization of ToolInvoker with a custom Toolset."""
        custom_toolset = CustomToolset(tools=[weather_tool], custom_attr="custom_value")
        invoker = ToolInvoker(tools=custom_toolset)
        data = invoker.to_dict()

        assert isinstance(data, dict)
        assert "type" in data and "init_parameters" in data
        tools_data = data["init_parameters"]["tools"]
        assert isinstance(tools_data, dict)
        assert len(tools_data["data"]["tools"]) == 1
        assert tools_data["data"]["tools"][0]["type"] == "haystack.tools.tool.Tool"
        assert tools_data.get("custom_attr") == "custom_value"

        deserialized_invoker = ToolInvoker.from_dict(data)
        assert deserialized_invoker.tools == invoker.tools
        assert deserialized_invoker._tools_with_names == invoker._tools_with_names
        assert deserialized_invoker.raise_on_failure == invoker.raise_on_failure
        assert deserialized_invoker.convert_result_to_json_string == invoker.convert_result_to_json_string
