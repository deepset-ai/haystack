# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset, tool
from haystack.tools.toolset import _ToolsetWrapper


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


@tool
def rebuilt() -> str:
    """A rebuilt tool."""
    return "rebuilt"


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


class RebuildingToolset(Toolset):
    """A toolset that rebuilds its tools on from_dict() instead of serializing them (like a dynamic toolset)."""

    def __init__(self):
        super().__init__([rebuilt])

    def to_dict(self):
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data):
        return cls()


class TestToolsetWrapper:
    """Tests for the _ToolsetWrapper class"""

    def test_toolset_plus_toolset_creates_wrapper(self, add_tool, multiply_tool):
        """Test that combining two Toolsets creates a _ToolsetWrapper and works correctly."""
        result = Toolset([add_tool]) + Toolset([multiply_tool])
        assert isinstance(result, _ToolsetWrapper)
        assert len(result) == 2
        assert add_tool in result
        assert multiply_tool in result

    def test_wrapper_with_agent(self, add_tool, multiply_tool, monkeypatch):
        """Test that _ToolsetWrapper works with Agent."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        wrapper = Toolset([add_tool]) + Toolset([multiply_tool])
        agent = Agent(chat_generator=OpenAIChatGenerator(), tools=wrapper)
        agent.warm_up()
        assert len(list(agent.tools)) == 2

    def test_wrapper_chaining_and_duplicate_detection(self, add_tool, multiply_tool, subtract_tool):
        """Test chaining operations and that duplicates are still detected."""
        # Chaining should work
        result = Toolset([add_tool]) + Toolset([multiply_tool]) + Toolset([subtract_tool])
        assert len(result) == 3
        # Duplicates should be detected
        toolset_with_dup = Toolset([add_tool])
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            _ = result + toolset_with_dup


class TestToolsetWrapperWarmUp:
    """Tests for warm_up behavior of _ToolsetWrapper."""

    def test_new_wrapper_is_not_warmed_up(self):
        wrapper = Toolset([WarmUpCountingTool("a")]) + Toolset([WarmUpCountingTool("b")])
        assert wrapper._is_warmed_up is False

    def test_warm_up_delegates_to_each_toolset(self):
        ts1 = WarmUpCountingToolset([WarmUpCountingTool("a")])
        ts2 = WarmUpCountingToolset([WarmUpCountingTool("b")])
        wrapper = ts1 + ts2
        wrapper.warm_up()
        assert ts1.warm_up_count == 1
        assert ts2.warm_up_count == 1
        assert wrapper._is_warmed_up is True

    def test_warm_up_is_idempotent(self):
        ts1 = WarmUpCountingToolset([WarmUpCountingTool("a")])
        ts2 = WarmUpCountingToolset([WarmUpCountingTool("b")])
        wrapper = ts1 + ts2
        wrapper.warm_up()
        wrapper.warm_up()
        wrapper.warm_up()
        assert ts1.warm_up_count == 1
        assert ts2.warm_up_count == 1


class TestToolsetWrapperSerialization:
    """Tests for to_dict/from_dict of _ToolsetWrapper."""

    def test_to_dict(self, add_tool, multiply_tool):
        wrapper = Toolset([add_tool]) + Toolset([multiply_tool])
        data = wrapper.to_dict()
        assert data["type"] == "haystack.tools.toolset._ToolsetWrapper"
        assert len(data["data"]["toolsets"]) == 2
        assert all(ts["type"] == "haystack.tools.toolset.Toolset" for ts in data["data"]["toolsets"])

    def test_from_dict_round_trip(self, add_tool, multiply_tool):
        wrapper = Toolset([add_tool]) + Toolset([multiply_tool])
        restored = _ToolsetWrapper.from_dict(wrapper.to_dict())
        assert isinstance(restored, _ToolsetWrapper)
        assert len(restored) == 2
        assert len(restored.toolsets) == 2
        assert "add" in restored
        assert "multiply" in restored

    def test_to_dict_preserves_subclass_serialization(self, add_tool):
        # RebuildingToolset has a custom to_dict that serializes no tools (they are rebuilt on from_dict).
        wrapper = RebuildingToolset() + Toolset([add_tool])
        data = wrapper.to_dict()

        # Each wrapped toolset is serialized via its own to_dict, so the custom one is preserved.
        assert data["data"]["toolsets"][0]["type"].endswith("RebuildingToolset")
        assert data["data"]["toolsets"][0]["data"] == {}

        restored = _ToolsetWrapper.from_dict(data)
        assert isinstance(restored.toolsets[0], RebuildingToolset)
        assert "rebuilt" in restored
        assert "add" in restored

    def test_from_dict_rejects_non_toolset(self, add_tool):
        data = Toolset([add_tool]).to_dict()
        data["data"] = {"toolsets": [{"type": "haystack.tools.tool.Tool", "data": {}}]}

        with pytest.raises(TypeError, match="is not a subclass of Toolset"):
            _ToolsetWrapper.from_dict(data)
