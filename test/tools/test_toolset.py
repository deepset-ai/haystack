# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.converters import OutputAdapter
from haystack.dataclasses.chat_message import ToolCall
from haystack.tools import Tool, Toolset


def test_toolset_with_tool_invoker():
    """Test that a Toolset works with ToolInvoker."""

    # Define a simple function for a tool
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # Create a tool
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

    # Create a toolset with the tool
    toolset = Toolset([add_tool])

    # Create a ToolInvoker with the toolset
    invoker = ToolInvoker(tools=toolset)

    # Create a message with a tool call
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])

    # Invoke the tool
    result = invoker.run(messages=[message])

    # Check that the tool was invoked correctly
    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_toolset_with_multiple_tools():
    """Test that a Toolset with multiple tools works properly."""

    # Define simple functions for tools
    def add_numbers(a: int, b: int) -> int:
        return a + b

    def multiply_numbers(a: int, b: int) -> int:
        return a * b

    # Create tools
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

    # Create a toolset with the tools
    toolset = Toolset([add_tool, multiply_tool])

    # Check that both tools are in the toolset
    assert len(toolset) == 2
    assert toolset[0].name == "add"
    assert toolset[1].name == "multiply"

    # Create a ToolInvoker with the toolset
    invoker = ToolInvoker(tools=list(toolset))

    # Create messages with tool calls
    add_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    add_message = ChatMessage.from_assistant(tool_calls=[add_call])

    multiply_call = ToolCall(tool_name="multiply", arguments={"a": 4, "b": 5})
    multiply_message = ChatMessage.from_assistant(tool_calls=[multiply_call])

    # Invoke the tools
    result = invoker.run(messages=[add_message, multiply_message])

    # Check that the tools were invoked correctly
    assert len(result["tool_messages"]) == 2
    tool_results = [message.tool_call_result.result for message in result["tool_messages"]]
    assert "5" in tool_results
    assert "20" in tool_results


def test_toolset_adding():
    """Test that tools can be added to a Toolset."""

    # Create an empty toolset
    toolset = Toolset()
    assert len(toolset) == 0

    # Define a simple function for a tool
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # Create a tool
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

    # Add the tool
    toolset.add(add_tool)
    assert len(toolset) == 1
    assert toolset[0].name == "add"

    # Test with ToolInvoker
    invoker = ToolInvoker(tools=list(toolset))
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])
    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_toolset_addition():
    """Test that toolsets can be combined."""

    # Define simple functions for tools
    def add_numbers(a: int, b: int) -> int:
        return a + b

    def multiply_numbers(a: int, b: int) -> int:
        return a * b

    def subtract_numbers(a: int, b: int) -> int:
        return a - b

    # Create tools
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

    # Create toolsets
    toolset1 = Toolset([add_tool])
    toolset2 = Toolset([multiply_tool])

    # Combine toolsets
    combined_toolset = toolset1 + toolset2
    assert len(combined_toolset) == 2

    # Add a single tool
    combined_toolset = combined_toolset + subtract_tool
    assert len(combined_toolset) == 3

    # Check that all tools are accessible
    tool_names = [tool.name for tool in combined_toolset]
    assert "add" in tool_names
    assert "multiply" in tool_names
    assert "subtract" in tool_names

    # Test with ToolInvoker
    invoker = ToolInvoker(tools=list(combined_toolset))

    # Create messages with tool calls
    add_call = ToolCall(tool_name="add", arguments={"a": 10, "b": 5})
    multiply_call = ToolCall(tool_name="multiply", arguments={"a": 10, "b": 5})
    subtract_call = ToolCall(tool_name="subtract", arguments={"a": 10, "b": 5})

    message = ChatMessage.from_assistant(tool_calls=[add_call, multiply_call, subtract_call])

    # Invoke the tools
    result = invoker.run(messages=[message])

    # Check that all tools were invoked correctly
    assert len(result["tool_messages"]) == 3
    tool_results = [message.tool_call_result.result for message in result["tool_messages"]]
    assert "15" in tool_results
    assert "50" in tool_results
    assert "5" in tool_results


def add_numbers_for_serialization(a: int, b: int) -> int:
    return a + b


def test_toolset_serialization():
    """Test that a Toolset can be serialized and deserialized."""

    # Create a tool
    add_tool = Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers_for_serialization,
    )

    # Create a toolset with the tool
    toolset = Toolset([add_tool])

    # Serialize the toolset
    serialized = toolset.to_dict()

    # Deserialize the toolset
    deserialized = Toolset.from_dict(serialized)

    # Check that the deserialized toolset has the same tool
    assert len(deserialized) == 1
    assert deserialized[0].name == "add"
    assert deserialized[0].description == "Add two numbers"

    # Test that the deserialized toolset works with ToolInvoker
    invoker = ToolInvoker(tools=list(deserialized))
    tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    message = ChatMessage.from_assistant(tool_calls=[tool_call])
    result = invoker.run(messages=[message])

    assert len(result["tool_messages"]) == 1
    assert result["tool_messages"][0].tool_call_result.result == "5"


def test_toolset_duplicate_tool_names():
    """Test that a Toolset raises an error for duplicate tool names."""

    # Define a simple function for tools
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # Create tools with the same name
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

    # Check that creating a toolset with duplicate tool names raises an error
    with pytest.raises(ValueError, match="Duplicate tool names found"):
        Toolset([add_tool1, add_tool2])

    # Create a toolset with one tool
    toolset = Toolset([add_tool1])

    # Check that adding a tool with a duplicate name raises an error
    with pytest.raises(ValueError, match="Duplicate tool names found"):
        toolset.add(add_tool2)

    # Check that combining toolsets with duplicate tool names raises an error
    toolset2 = Toolset([add_tool2])
    with pytest.raises(ValueError, match="Duplicate tool names found"):
        combined = toolset + toolset2


def add_numbers_for_pipeline(a: int, b: int) -> int:
    return a + b


def multiply_numbers_for_pipeline(a: int, b: int) -> int:
    return a * b


def add_numbers_for_calculator(a: int, b: int) -> int:
    return a + b


# Integration tests using full pipelines
@pytest.mark.integration
class TestToolsetIntegration:
    """Integration tests for Toolset in complete pipelines."""

    def test_basic_math_pipeline(self):
        """Test basic math operations through a complete pipeline flow."""

        def add_numbers(a: int, b: int) -> int:
            return a + b

        def subtract_numbers(a: int, b: int) -> int:
            return a - b

        # Create the tools
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

        # Create a toolset
        math_toolset = Toolset([add_tool, subtract_tool])

        # Create a complete pipeline
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=list(math_toolset)))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=list(math_toolset)))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        # Connect the components
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        # Test addition through the pipeline
        user_input = "What is 5 plus 3?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        # Verify the complete flow worked
        pipe_result = result["response_llm"]["replies"][0].text
        assert "8" in pipe_result or "eight" in pipe_result

    def test_combined_toolsets_pipeline(self):
        """Test combining multiple toolsets in a complete pipeline."""

        # Create basic and advanced math tools
        def add_numbers(a: int, b: int) -> int:
            return a + b

        def multiply_numbers(a: int, b: int) -> int:
            return a * b

        # Create the tools
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

        # Create separate toolsets
        basic_math_toolset = Toolset([add_tool])
        advanced_math_toolset = Toolset([multiply_tool])

        # Combine toolsets
        combined_toolset = basic_math_toolset + advanced_math_toolset

        # Create a complete pipeline
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=combined_toolset))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=list(combined_toolset)))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        # Connect the components
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        # Test multiplication through the pipeline
        user_input = "What is 6 times 7?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        # Verify the complete flow worked
        pipe_result = result["response_llm"]["replies"][0].text
        assert "42" in pipe_result or "forty two" in pipe_result

    def test_custom_calculator_pipeline(self):
        """Test a custom calculator toolset in a complete pipeline."""

        class CalculatorToolset(Toolset):
            """A toolset for calculator operations."""

            def __init__(self):
                """Create a toolset with calculator operations."""
                super().__init__([])
                self._create_tools()

            def _create_tools(self):
                """Create calculator operation tools."""

                def add_numbers(a: int, b: int) -> int:
                    return a + b

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

                def multiply_numbers(a: int, b: int) -> int:
                    return a * b

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

                # Add the tools
                self.add(add_tool)
                self.add(multiply_tool)

        # Create a CalculatorToolset instance
        calculator_toolset = CalculatorToolset()

        # Create a complete pipeline
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=list(calculator_toolset)))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=list(calculator_toolset)))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-4o-mini"))

        # Connect the components
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        # Test the calculator through the pipeline
        user_input = "What is 15 plus 7?"
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        # Verify the complete flow worked
        pipe_result = result["response_llm"]["replies"][0].text
        assert "22" in pipe_result or "twenty two" in pipe_result

    def test_serde_in_pipeline(self, monkeypatch):
        """Test serialization and deserialization of a pipeline with toolsets."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create tools
        add_tool = Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=add_numbers_for_pipeline,
        )

        multiply_tool = Tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=multiply_numbers_for_pipeline,
        )

        # Create toolsets and combine them
        basic_toolset = Toolset([add_tool])
        advanced_toolset = Toolset([multiply_tool])
        combined_toolset = basic_toolset + advanced_toolset

        # Create and configure the pipeline
        pipeline = Pipeline()
        pipeline.add_component("tool_invoker", ToolInvoker(tools=list(combined_toolset)))
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo", tools=list(combined_toolset)))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        # Serialize to dict and verify structure
        pipeline_dict = pipeline.to_dict()
        assert (
            pipeline_dict["components"]["tool_invoker"]["type"] == "haystack.components.tools.tool_invoker.ToolInvoker"
        )
        assert len(pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]) == 2

        # Verify tool serialization
        tools_dict = pipeline_dict["components"]["tool_invoker"]["init_parameters"]["tools"]
        tool_names = [tool["data"]["name"] for tool in tools_dict]
        assert "add" in tool_names
        assert "multiply" in tool_names

        # Test round-trip serialization
        pipeline_dict = pipeline.to_dict()
        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline == pipeline

    def test_toolset_serde_in_pipeline(self):
        """Test serialization and deserialization of toolsets within a pipeline."""

        class CalculatorToolset(Toolset):
            """A toolset for calculator operations."""

            def __init__(self):
                """Create a toolset with calculator operations."""
                super().__init__([])
                self._create_tools()

            def _create_tools(self):
                """Create calculator operation tools."""
                add_tool = Tool(
                    name="add",
                    description="Add two numbers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                    function=add_numbers_for_calculator,
                )

                self.add(add_tool)

        # Create a CalculatorToolset instance
        calculator_toolset = CalculatorToolset()

        # Create a pipeline with the toolset
        pipeline = Pipeline()
        pipeline.add_component("tool_invoker", ToolInvoker(tools=list(calculator_toolset)))

        # Serialize the pipeline
        pipeline_dict = pipeline.to_dict()

        # Verify toolset serialization
        tool_invoker_dict = pipeline_dict["components"]["tool_invoker"]
        assert tool_invoker_dict["type"] == "haystack.components.tools.tool_invoker.ToolInvoker"
        assert len(tool_invoker_dict["init_parameters"]["tools"]) == 1

        # Verify tool details
        tool_dict = tool_invoker_dict["init_parameters"]["tools"][0]
        assert tool_dict["data"]["name"] == "add"
        assert tool_dict["data"]["description"] == "Add two numbers"

        # Test round-trip serialization
        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline
