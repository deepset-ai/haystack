# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents import Agent
from haystack.components.tools import ToolInvoker
from haystack.core.component.component import component
from haystack.tools import Tool, Toolset, warm_up_tools
from haystack.tools.toolset import _ToolsetWrapper


# Test fixtures
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def subtract_numbers(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


class WarmupTrackingToolset(Toolset):
    """A toolset that tracks whether warm_up was called."""

    def __init__(self, tools):
        super().__init__(tools)
        self.was_warmed_up = False

    def warm_up(self):
        self.was_warmed_up = True


@pytest.fixture
def add_tool():
    return Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=add_numbers,
    )


@pytest.fixture
def multiply_tool():
    return Tool(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=multiply_numbers,
    )


@pytest.fixture
def subtract_tool():
    return Tool(
        name="subtract",
        description="Subtract two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        function=subtract_numbers,
    )


class TestToolsetWrapper:
    """Tests for the _ToolsetWrapper class"""

    def test_toolset_plus_toolset_creates_wrapper(self, add_tool, multiply_tool):
        """Test that combining two Toolsets creates a _ToolsetWrapper and works correctly."""
        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool])

        result = toolset1 + toolset2

        assert isinstance(result, _ToolsetWrapper)
        assert len(result) == 2
        assert add_tool in result
        assert multiply_tool in result

    def test_wrapper_warm_up_calls_each_toolset(self, add_tool, multiply_tool):
        """Test that warming up a wrapper calls warm_up on each toolset - the main reason for wrapper."""
        toolset1 = WarmupTrackingToolset([add_tool])
        toolset2 = WarmupTrackingToolset([multiply_tool])

        wrapper = toolset1 + toolset2
        wrapper.warm_up()

        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up

    def test_wrapper_with_tool_invoker(self, add_tool, multiply_tool):
        """Test that _ToolsetWrapper works with ToolInvoker and tools are warmed up."""
        toolset1 = WarmupTrackingToolset([add_tool])
        toolset2 = WarmupTrackingToolset([multiply_tool])
        wrapper = toolset1 + toolset2

        # ToolInvoker should warm up tools when warm_up is called
        invoker = ToolInvoker(tools=wrapper)
        invoker.warm_up()

        assert len(invoker._tools_with_names) == 2
        assert "add" in invoker._tools_with_names
        assert "multiply" in invoker._tools_with_names
        # Verify warmup was called on each toolset
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up

    def test_wrapper_with_agent(self, add_tool, multiply_tool):
        """Test that _ToolsetWrapper works with Agent."""

        @component
        class MockChatGenerator:
            def run(self, messages, tools=None, **kwargs):
                return {"replies": messages}

            def warm_up(self):
                pass

        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool])
        wrapper = toolset1 + toolset2

        agent = Agent(chat_generator=MockChatGenerator(), tools=wrapper)
        agent.warm_up()

        assert len(list(agent.tools)) == 2

    def test_wrapper_chaining_and_duplicate_detection(self, add_tool, multiply_tool, subtract_tool):
        """Test chaining operations and that duplicates are still detected."""
        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool])
        toolset3 = Toolset([subtract_tool])

        # Chaining should work
        result = toolset1 + toolset2 + toolset3
        assert len(result) == 3

        # Duplicates should be detected
        toolset_with_dup = Toolset([add_tool])
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            _ = result + toolset_with_dup
