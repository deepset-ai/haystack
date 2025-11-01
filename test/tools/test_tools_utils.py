# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, Toolset, flatten_tools_or_toolsets, warm_up_tools


def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def subtract_numbers(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


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


class TestFlattenToolsOrToolsets:
    def test_flatten_none(self):
        """Test that None returns an empty list."""
        result = flatten_tools_or_toolsets(None)
        assert result == []

    def test_flatten_empty_list(self):
        """Test that an empty list returns an empty list."""
        result = flatten_tools_or_toolsets([])
        assert result == []

    def test_flatten_list_of_tools(self, add_tool, multiply_tool):
        """Test that a list of Tool instances is returned as-is."""
        tools = [add_tool, multiply_tool]
        result = flatten_tools_or_toolsets(tools)
        assert result == tools
        assert len(result) == 2
        assert result[0].name == "add"
        assert result[1].name == "multiply"

    def test_flatten_single_toolset(self, add_tool, multiply_tool):
        """Test that a single Toolset is converted to a list of Tools."""
        toolset = Toolset([add_tool, multiply_tool])
        result = flatten_tools_or_toolsets(toolset)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, Tool) for t in result)
        assert result[0].name == "add"
        assert result[1].name == "multiply"

    def test_flatten_list_of_toolsets(self, add_tool, multiply_tool, subtract_tool):
        """Test that a list of Toolset instances is flattened to a single list of Tools."""
        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool, subtract_tool])

        result = flatten_tools_or_toolsets([toolset1, toolset2])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(t, Tool) for t in result)
        assert result[0].name == "add"
        assert result[1].name == "multiply"
        assert result[2].name == "subtract"

    def test_flatten_list_with_mixed_tools_and_toolsets(self, add_tool, multiply_tool, subtract_tool):
        """Test that a mixed list of Tool and Toolset instances is flattened correctly."""
        toolset = Toolset([multiply_tool])
        mixed_list = [add_tool, toolset, subtract_tool]

        result = flatten_tools_or_toolsets(mixed_list)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(t, Tool) for t in result)
        assert result[0].name == "add"
        assert result[1].name == "multiply"
        assert result[2].name == "subtract"

    def test_flatten_empty_toolset(self):
        """Test that an empty Toolset returns an empty list."""
        toolset = Toolset([])
        result = flatten_tools_or_toolsets(toolset)
        assert result == []

    def test_flatten_list_with_empty_toolsets(self, add_tool):
        """Test that a list with empty Toolsets handles correctly."""
        toolset1 = Toolset([])
        toolset2 = Toolset([add_tool])
        toolset3 = Toolset([])

        result = flatten_tools_or_toolsets([toolset1, toolset2, toolset3])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "add"

    def test_flatten_invalid_type_in_list(self):
        """Test that invalid types in the list raise TypeError."""
        with pytest.raises(TypeError, match="Items in the tools list must be Tool or Toolset instances"):
            flatten_tools_or_toolsets(["not_a_tool"])

        with pytest.raises(TypeError, match="Items in the tools list must be Tool or Toolset instances"):
            flatten_tools_or_toolsets([123])

        with pytest.raises(TypeError, match="Items in the tools list must be Tool or Toolset instances"):
            flatten_tools_or_toolsets([{"key": "value"}])

    def test_flatten_invalid_type(self):
        """Test that invalid root types raise TypeError."""
        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets("not_valid")

        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets(123)

        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets({"key": "value"})

    def test_flatten_multiple_toolsets(self, add_tool, multiply_tool, subtract_tool):
        """Test flattening a list of multiple Toolsets."""
        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool])
        toolset3 = Toolset([subtract_tool])

        # List of three separate toolsets
        result = flatten_tools_or_toolsets([toolset1, toolset2, toolset3])
        assert len(result) == 3
        assert result[0].name == "add"
        assert result[1].name == "multiply"
        assert result[2].name == "subtract"


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


class TestWarmUpTools:
    """Tests for the warm_up_tools() function"""

    def test_warm_up_tools_with_none(self):
        """Test that warm_up_tools with None does nothing."""
        # Should not raise any errors
        warm_up_tools(None)

    def test_warm_up_tools_with_single_tool(self):
        """Test that warm_up_tools works with a single tool in a list."""
        tool = WarmupTrackingTool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test",
        )

        assert not tool.was_warmed_up
        warm_up_tools([tool])
        assert tool.was_warmed_up

    def test_warm_up_tools_with_single_toolset(self):
        """
        Test that when passing a single Toolset, both the Toolset.warm_up()
        and each individual tool's warm_up() are called.
        """
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

        toolset = WarmupTrackingToolset([tool1, tool2])

        assert not toolset.was_warmed_up
        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up

        warm_up_tools(toolset)

        # Both the toolset itself and individual tools should be warmed up
        assert toolset.was_warmed_up
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up

    def test_warm_up_tools_with_list_containing_toolset(self):
        """Test that when a Toolset is in a list, individual tools inside get warmed up."""
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

        toolset = WarmupTrackingToolset([tool1, tool2])

        assert not toolset.was_warmed_up
        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up

        warm_up_tools([toolset])

        # Both the toolset itself and individual tools should be warmed up
        assert toolset.was_warmed_up
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up

    def test_warm_up_tools_with_multiple_toolsets(self):
        """Test multiple Toolsets in a list."""
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
        tool3 = WarmupTrackingTool(
            name="tool3",
            description="Third tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "tool3",
        )

        toolset1 = WarmupTrackingToolset([tool1])
        toolset2 = WarmupTrackingToolset([tool2, tool3])

        assert not toolset1.was_warmed_up
        assert not toolset2.was_warmed_up
        assert not tool1.was_warmed_up
        assert not tool2.was_warmed_up
        assert not tool3.was_warmed_up

        warm_up_tools([toolset1, toolset2])

        # Both toolsets and all individual tools should be warmed up
        assert toolset1.was_warmed_up
        assert toolset2.was_warmed_up
        assert tool1.was_warmed_up
        assert tool2.was_warmed_up
        assert tool3.was_warmed_up

    def test_warm_up_tools_with_mixed_tools_and_toolsets(self):
        """Test list with both Tool objects and Toolsets."""
        standalone_tool = WarmupTrackingTool(
            name="standalone",
            description="Standalone tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "standalone",
        )
        toolset_tool1 = WarmupTrackingTool(
            name="toolset_tool1",
            description="Tool in toolset",
            parameters={"type": "object", "properties": {}},
            function=lambda: "toolset_tool1",
        )
        toolset_tool2 = WarmupTrackingTool(
            name="toolset_tool2",
            description="Another tool in toolset",
            parameters={"type": "object", "properties": {}},
            function=lambda: "toolset_tool2",
        )

        toolset = WarmupTrackingToolset([toolset_tool1, toolset_tool2])

        assert not standalone_tool.was_warmed_up
        assert not toolset.was_warmed_up
        assert not toolset_tool1.was_warmed_up
        assert not toolset_tool2.was_warmed_up

        warm_up_tools([standalone_tool, toolset])

        # All tools and the toolset should be warmed up
        assert standalone_tool.was_warmed_up
        assert toolset.was_warmed_up
        assert toolset_tool1.was_warmed_up
        assert toolset_tool2.was_warmed_up

    def test_warm_up_tools_idempotency(self):
        """Test that calling warm_up_tools() multiple times is safe."""

        class WarmupCountingTool(Tool):
            """A tool that counts how many times warm_up was called."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.warm_up_count = 0

            def warm_up(self):
                self.warm_up_count += 1

        class WarmupCountingToolset(Toolset):
            """A toolset that counts how many times warm_up was called."""

            def __init__(self, tools):
                super().__init__(tools)
                self.warm_up_count = 0

            def warm_up(self):
                self.warm_up_count += 1

        tool = WarmupCountingTool(
            name="counting_tool",
            description="A counting tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test",
        )
        toolset = WarmupCountingToolset([tool])

        # Call warm_up_tools multiple times
        warm_up_tools(toolset)
        warm_up_tools(toolset)
        warm_up_tools(toolset)

        # warm_up_tools itself doesn't prevent multiple calls,
        # but verify the calls actually happen multiple times
        assert toolset.warm_up_count == 3
        assert tool.warm_up_count == 3
