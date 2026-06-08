# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents import Agent
from haystack.core.component.component import component
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset
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
        super().__init__(
            [
                Tool(
                    name="rebuilt",
                    description="A rebuilt tool",
                    parameters={"type": "object", "properties": {}},
                    function=add_numbers,
                )
            ]
        )

    def to_dict(self):
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data):
        return cls()


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
