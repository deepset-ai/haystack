# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, Toolset, flatten_tools_or_toolsets


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

    def test_flatten_nested_toolsets(self, add_tool, multiply_tool, subtract_tool):
        """Test flattening multiple levels of Toolsets."""
        toolset1 = Toolset([add_tool])
        toolset2 = Toolset([multiply_tool])
        toolset3 = Toolset([subtract_tool])

        # List of three separate toolsets
        result = flatten_tools_or_toolsets([toolset1, toolset2, toolset3])
        assert len(result) == 3
        assert result[0].name == "add"
        assert result[1].name == "multiply"
        assert result[2].name == "subtract"
