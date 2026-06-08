# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

import pytest

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.agents.tool_calling import _run_tool
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ToolCall
from haystack.tools import Tool, Toolset, tool
from haystack.tools.errors import ToolInvocationError


@component
class MockChatGenerator:
    def to_dict(self):
        return {"type": "test_toolset.MockChatGenerator", "init_parameters": {}}

    @classmethod
    def from_dict(cls, data):
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs: Any
    ) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("done")]}


def _run_tool_messages(messages: list[ChatMessage], tools: Toolset | list[Tool | Toolset]) -> list[ChatMessage]:
    tool_messages, _ = _run_tool(messages=messages, state=State(schema={}), tools=tools)
    return tool_messages


class DynamicToolset(Toolset):
    """A custom Toolset that recreates its tools dynamically on deserialization instead of serializing them."""

    def __init__(self):
        super().__init__([add])

    def to_dict(self):
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data):
        return cls()


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


# Defined at module level (not inside the fixtures) so the underlying functions are importable and serializable.
@tool
def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: Annotated[int, "first number"], b: Annotated[int, "second number"]) -> int:
    """Subtract b from a."""
    return a - b


@pytest.fixture
def add_tool():
    return add


@pytest.fixture
def multiply_tool():
    return multiply


@pytest.fixture
def subtract_tool():
    return subtract


class WarmUpCountingTool(Tool):
    """A Tool that records how many times warm_up() was called."""

    def __init__(self, name: str):
        super().__init__(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: None,
        )
        self.warm_up_count = 0

    def warm_up(self) -> None:
        self.warm_up_count += 1


class WarmUpCountingToolset(Toolset):
    """A Toolset that records how many times its own warm_up() did real work."""

    def __init__(self, tools):
        super().__init__(tools)
        self.warm_up_count = 0

    def warm_up(self) -> None:
        if self._is_warmed_up:
            return
        self.warm_up_count += 1
        super().warm_up()


class TestToolset:
    def test_toolset_with_multiple_tools(self, add_tool, multiply_tool):
        """Test that a Toolset with multiple tools works properly."""
        toolset = Toolset([add_tool, multiply_tool])

        assert len(toolset) == 2
        assert toolset[0].name == "add"
        assert toolset[1].name == "multiply"

        add_message = ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="add", arguments={"a": 2, "b": 3})])
        multiply_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="multiply", arguments={"a": 4, "b": 5})]
        )
        tool_messages = _run_tool_messages(messages=[add_message, multiply_message], tools=toolset)

        assert len(tool_messages) == 2
        tool_results = [tcr.result for message in tool_messages for tcr in message.tool_call_results]
        assert "5" in tool_results
        assert "20" in tool_results

    def test_toolset_add(self, add_tool):
        """Test that tools can be added to a Toolset."""
        toolset = Toolset()
        assert len(toolset) == 0

        toolset.add(add_tool)
        assert len(toolset) == 1
        assert toolset[0].name == "add"

        message = ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="add", arguments={"a": 2, "b": 3})])
        tool_messages = _run_tool_messages(messages=[message], tools=toolset)

        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_results[0].result == "5"

    def test_toolset_contains(self, add_tool, multiply_tool):
        """Test that the __contains__ method works correctly."""
        toolset = Toolset([add_tool])
        # Test with a tool instance
        assert add_tool in toolset
        assert multiply_tool not in toolset
        # Test with a tool name
        assert "add" in toolset
        assert "multiply" not in toolset
        assert "non_existent_tool" not in toolset

    def test_toolset_addition(self, add_tool, multiply_tool, subtract_tool):
        """Test that toolsets can be combined."""
        combined_toolset = Toolset([add_tool]) + Toolset([multiply_tool])
        assert len(combined_toolset) == 2
        assert isinstance(combined_toolset, Toolset)

        combined_toolset = combined_toolset + subtract_tool
        assert len(combined_toolset) == 3
        assert isinstance(combined_toolset, Toolset)

        tool_names = [t.name for t in combined_toolset]
        assert "add" in tool_names
        assert "multiply" in tool_names
        assert "subtract" in tool_names

        add_call = ToolCall(tool_name="add", arguments={"a": 10, "b": 5})
        multiply_call = ToolCall(tool_name="multiply", arguments={"a": 10, "b": 5})
        subtract_call = ToolCall(tool_name="subtract", arguments={"a": 10, "b": 5})
        message = ChatMessage.from_assistant(tool_calls=[add_call, multiply_call, subtract_call])
        tool_messages = _run_tool_messages(messages=[message], tools=combined_toolset)

        assert len(tool_messages) == 3
        tool_results = [tcr.result for message in tool_messages for tcr in message.tool_call_results]
        assert "15" in tool_results
        assert "50" in tool_results
        assert "5" in tool_results

    def test_toolset_add_various_types(self, add_tool, multiply_tool, subtract_tool):
        """Test that the __add__ method works with various object types."""
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
            toolset1 + "not_a_tool"  # type: ignore[operator]

        with pytest.raises(TypeError):
            toolset1 + 123  # type: ignore[operator]

    def test_toolset_serialization(self, add_tool):
        """Test that a Toolset can be serialized and deserialized."""
        serialized = Toolset([add_tool]).to_dict()
        deserialized = Toolset.from_dict(serialized)

        assert len(deserialized) == 1
        assert deserialized[0].name == "add"
        assert deserialized[0].description == "Add two numbers."

        tool_call = ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])
        tool_messages = _run_tool_messages(messages=[message], tools=deserialized)

        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_results[0].result == "5"

    def test_toolset_duplicate_tool_names(self, add_tool):
        """Test that a Toolset raises an error for duplicate tool names."""
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            Toolset([add_tool, add_tool])

        toolset = Toolset([add_tool])

        with pytest.raises(ValueError, match="Duplicate tool names found"):
            toolset.add(add_tool)

        toolset2 = Toolset([add_tool])
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            _ = toolset + toolset2


class TestToolsetWithAgent:
    def test_init_with_toolset(self, weather_tool):
        """Test initializing Agent with a Toolset."""
        toolset = Toolset(tools=[weather_tool])
        agent = Agent(chat_generator=MockChatGenerator(), tools=toolset)
        assert agent.tools == toolset

    def test_tool_invocation_error_with_toolset(self, faulty_tool):
        """Test tool invocation errors with a Toolset."""
        toolset = Toolset(tools=[faulty_tool])
        tool_call = ToolCall(tool_name="faulty_tool", arguments={"location": "Berlin"})
        tool_call_message = ChatMessage.from_assistant(tool_calls=[tool_call])
        with pytest.raises(ToolInvocationError):
            _run_tool(messages=[tool_call_message], state=State(schema={}), tools=toolset)

    def test_custom_toolset_serde_in_agent(self):
        """Test serialization and deserialization of a custom toolset within an Agent."""
        agent = Agent(chat_generator=MockChatGenerator(), tools=DynamicToolset())
        agent_dict = agent.to_dict()
        tools_dict = agent_dict["init_parameters"]["tools"]
        assert tools_dict["type"] == "test_toolset.DynamicToolset"
        assert len(tools_dict["data"]) == 0
        new_agent = Agent.from_dict(agent_dict)
        assert isinstance(new_agent.tools, DynamicToolset)

    def test_serde_with_toolset(self, add_tool, multiply_tool):
        """Test serialization and deserialization of regular Toolsets within an Agent."""
        toolset = Toolset([add_tool, multiply_tool])
        agent = Agent(chat_generator=MockChatGenerator(), tools=toolset)
        agent_dict = agent.to_dict()
        tools_dict = agent_dict["init_parameters"]["tools"]
        assert tools_dict["type"] == "haystack.tools.toolset.Toolset"
        assert len(tools_dict["data"]["tools"]) == 2
        tool_names = [tool["data"]["name"] for tool in tools_dict["data"]["tools"]]
        assert "add" in tool_names
        assert "multiply" in tool_names
        new_agent = Agent.from_dict(agent_dict)
        assert isinstance(new_agent.tools, Toolset)
        assert [tool.name for tool in new_agent.tools] == ["add", "multiply"]

    def test_agent_serde_with_list_of_toolsets(self, weather_tool, add_tool):
        """Test serialization and deserialization of Agent with a list of Toolsets."""
        agent = Agent(chat_generator=MockChatGenerator(), tools=[Toolset([weather_tool]), Toolset([add_tool])])
        data = agent.to_dict()

        # Verify serialization preserves list[Toolset] structure
        tools_data = data["init_parameters"]["tools"]
        assert isinstance(tools_data, list)
        assert len(tools_data) == 2
        assert all(isinstance(ts, dict) for ts in tools_data)
        assert tools_data[0]["type"] == "haystack.tools.toolset.Toolset"
        assert tools_data[1]["type"] == "haystack.tools.toolset.Toolset"

        # Deserialize and verify
        deserialized_agent = Agent.from_dict(data)
        assert isinstance(deserialized_agent.tools, list)
        assert len(deserialized_agent.tools) == 2
        assert all(isinstance(ts, Toolset) for ts in deserialized_agent.tools)

    def test_list_of_toolsets_runtime_override(self, weather_tool, add_tool, multiply_tool):
        """Test that list of Toolsets can be passed as runtime override to Agent.run()."""
        toolset2 = Toolset([add_tool])
        toolset3 = Toolset([multiply_tool])

        @component
        class AddCallingChatGenerator:
            tool_invoked = False

            @component.output_types(replies=list[ChatMessage])
            def run(
                self, messages: list[ChatMessage], tools: list[Tool | Toolset] | None = None, **kwargs: Any
            ) -> dict[str, list[ChatMessage]]:
                assert tools == [toolset2, toolset3]
                if self.tool_invoked:
                    return {"replies": [ChatMessage.from_assistant("done")]}
                self.tool_invoked = True
                return {
                    "replies": [
                        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="add", arguments={"a": 3, "b": 7})])
                    ]
                }

        agent = Agent(chat_generator=AddCallingChatGenerator(), tools=Toolset([weather_tool]))
        result = agent.run(messages=[ChatMessage.from_user("Add numbers")], tools=[toolset2, toolset3])
        assert result["messages"][2].tool_call_result.result == "10"

    def test_pipeline_with_list_of_toolsets(self, add_tool, multiply_tool):
        """Test that a Pipeline can serialize/deserialize an Agent with a list of Toolsets."""
        pipeline = Pipeline()
        pipeline.add_component(
            "agent", Agent(chat_generator=MockChatGenerator(), tools=[Toolset([add_tool]), Toolset([multiply_tool])])
        )
        pipeline_dict = pipeline.to_dict()

        # Verify the serialized structure
        agent_dict = pipeline_dict["components"]["agent"]
        tools_data = agent_dict["init_parameters"]["tools"]
        assert isinstance(tools_data, list)
        assert len(tools_data) == 2
        assert all(ts["type"] == "haystack.tools.toolset.Toolset" for ts in tools_data)

        # Deserialize and verify functionality
        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline.to_dict() == pipeline_dict


class TestToolsetWarmUp:
    """Stress tests for Toolset warm_up behavior."""

    def test_new_toolset_is_not_warmed_up(self):
        toolset = Toolset([WarmUpCountingTool("a")])
        assert toolset._is_warmed_up is False

    def test_warm_up_warms_all_tools(self):
        t1, t2 = WarmUpCountingTool("a"), WarmUpCountingTool("b")
        toolset = Toolset([t1, t2])
        assert t1.warm_up_count == 0
        assert t2.warm_up_count == 0
        toolset.warm_up()
        assert t1.warm_up_count == 1
        assert t2.warm_up_count == 1
        assert toolset._is_warmed_up is True

    def test_warm_up_is_idempotent(self):
        t1 = WarmUpCountingTool("a")
        toolset = Toolset([t1])
        toolset.warm_up()
        toolset.warm_up()
        toolset.warm_up()
        assert t1.warm_up_count == 1

    def test_add_before_warm_up_does_not_warm_tools(self):
        existing = WarmUpCountingTool("a")
        toolset = Toolset([existing])
        new_tool = WarmUpCountingTool("b")
        toolset.add(new_tool)
        # Nothing is warmed until warm_up() is called explicitly.
        assert existing.warm_up_count == 0
        assert new_tool.warm_up_count == 0
        toolset.warm_up()
        assert existing.warm_up_count == 1
        assert new_tool.warm_up_count == 1

    def test_add_tool_after_warm_up_warms_only_new_tool(self):
        existing = WarmUpCountingTool("a")
        toolset = Toolset([existing])
        toolset.warm_up()
        assert existing.warm_up_count == 1
        new_tool = WarmUpCountingTool("b")
        toolset.add(new_tool)
        # The new tool is warmed immediately, the already-warmed tool is not re-warmed.
        assert new_tool.warm_up_count == 1
        assert existing.warm_up_count == 1

    def test_add_toolset_after_warm_up_warms_added_toolset(self):
        toolset = Toolset([WarmUpCountingTool("a")])
        toolset.warm_up()
        added_tools = [WarmUpCountingTool("b"), WarmUpCountingTool("c")]
        added = WarmUpCountingToolset(added_tools)
        toolset.add(added)
        # The added toolset's own warm_up() is invoked, warming its tools.
        assert added.warm_up_count == 1
        assert all(tool.warm_up_count == 1 for tool in added_tools)

    def test_plus_returns_new_unwarmed_toolset(self):
        ts1 = Toolset([WarmUpCountingTool("a")])
        ts1.warm_up()
        assert ts1._is_warmed_up is True
        new_tool = WarmUpCountingTool("b")
        ts2 = ts1 + new_tool
        # `+` returns a brand new Toolset object that has not been warmed up yet.
        assert ts2 is not ts1
        assert ts2._is_warmed_up is False
        assert new_tool.warm_up_count == 0
        ts2.warm_up()
        assert new_tool.warm_up_count == 1
