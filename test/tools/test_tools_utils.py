# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest

from haystack.tools import Tool, Toolset, flatten_tools_or_toolsets, warm_up_tools
from haystack.tools.utils import close_tools, close_tools_async, warm_up_tools_async


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
            flatten_tools_or_toolsets(["not_a_tool"])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="Items in the tools list must be Tool or Toolset instances"):
            flatten_tools_or_toolsets([123])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="Items in the tools list must be Tool or Toolset instances"):
            flatten_tools_or_toolsets([{"key": "value"}])  # type: ignore[list-item]

    def test_flatten_invalid_type(self):
        """Test that invalid root types raise TypeError."""
        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets("not_valid")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="tools must be list\\[Union\\[Tool, Toolset\\]\\], Toolset, or None"):
            flatten_tools_or_toolsets({"key": "value"})  # type: ignore[arg-type]

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


class TestWarmUpTools:
    def test_ignores_names_and_tools_without_warm_up(self, add_tool):
        warm_up_tools(None)
        warm_up_tools(["add"])
        warm_up_tools([add_tool])

    def test_warms_up_tools_in_mixed_list(self, add_tool, multiply_tool, subtract_tool, monkeypatch):
        add_warm_up, multiply_warm_up, subtract_warm_up = Mock(), Mock(), Mock()
        monkeypatch.setattr(add_tool, "warm_up", add_warm_up, raising=False)
        monkeypatch.setattr(multiply_tool, "warm_up", multiply_warm_up, raising=False)
        monkeypatch.setattr(subtract_tool, "warm_up", subtract_warm_up, raising=False)
        warm_up_tools([add_tool, Toolset([multiply_tool]), Toolset([subtract_tool])])
        add_warm_up.assert_called_once_with()
        multiply_warm_up.assert_called_once_with()
        subtract_warm_up.assert_called_once_with()

    def test_delegates_each_call_to_custom_toolset(self):
        tool = Mock(spec=Tool, warm_up=Mock())
        toolset = Mock(spec=Toolset, tools=[tool], warm_up=Mock())
        warm_up_tools(toolset)
        toolset.warm_up.assert_called_once_with()
        warm_up_tools(toolset)
        assert toolset.warm_up.call_count == 2
        tool.warm_up.assert_not_called()


class TestWarmUpToolsAsync:
    async def test_ignores_names_and_tools_without_warm_up(self, add_tool):
        await warm_up_tools_async(None)
        await warm_up_tools_async(["add"])
        await warm_up_tools_async([add_tool])

    async def test_prefers_async_warm_up_for_children(self):
        tool = Mock(spec=Tool, warm_up=Mock(), warm_up_async=AsyncMock())
        tool.name = "lookup"
        await warm_up_tools_async(Toolset([tool]))
        tool.warm_up_async.assert_awaited_once_with()
        tool.warm_up.assert_not_called()

    async def test_falls_back_to_sync_warm_up_for_children(self):
        tool = Mock(spec=Tool, warm_up=Mock())
        tool.name = "lookup"
        await warm_up_tools_async(Toolset([tool]))
        tool.warm_up.assert_called_once_with()

    async def test_falls_back_to_custom_toolset_warm_up(self):
        tool = Mock(spec=Tool, warm_up=Mock())
        toolset = Mock(spec=Toolset, tools=[tool], warm_up=Mock())
        await warm_up_tools_async(toolset)
        toolset.warm_up.assert_called_once_with()
        tool.warm_up.assert_not_called()


class TestCloseTools:
    def test_ignores_names_and_tools_without_close(self, add_tool):
        close_tools(None)
        close_tools(["add"])
        close_tools([add_tool])

    def test_closes_children_of_plain_toolset(self):
        tool = Mock(spec=Tool, close=Mock())
        tool.name = "lookup"
        close_tools(Toolset([tool]))
        tool.close.assert_called_once_with()

    def test_custom_toolset_owns_cleanup(self):
        tool = Mock(spec=Tool, close=Mock())
        toolset = Mock(spec=Toolset, tools=[tool], close=Mock())

        close_tools(toolset)

        toolset.close.assert_called_once_with()
        tool.close.assert_not_called()


class TestCloseToolsAsync:
    async def test_ignores_names_and_tools_without_close(self, add_tool):
        await close_tools_async(None)
        await close_tools_async(["add"])
        await close_tools_async([add_tool])

    async def test_prefers_async_close_for_children(self):
        tool = Mock(spec=Tool, close=Mock(), close_async=AsyncMock())
        tool.name = "lookup"
        await close_tools_async(Toolset([tool]))
        tool.close_async.assert_awaited_once_with()
        tool.close.assert_not_called()

    async def test_falls_back_to_custom_toolset_close(self):
        tool = Mock(spec=Tool, close=Mock())
        toolset = Mock(spec=Toolset, tools=[tool], close=Mock())
        await close_tools_async(toolset)
        toolset.close.assert_called_once_with()
        tool.close.assert_not_called()
