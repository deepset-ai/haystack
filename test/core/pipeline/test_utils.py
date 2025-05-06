# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import pytest

from haystack.core.pipeline.utils import parse_connect_string, FIFOPriorityQueue, deepcopy_with_fallback


def test_parse_connection():
    assert parse_connect_string("foobar") == ("foobar", None)
    assert parse_connect_string("foo.bar") == ("foo", "bar")


@pytest.fixture
def empty_queue():
    """Fixture providing a fresh empty queue for each test."""
    return FIFOPriorityQueue()


def test_empty_queue_initialization(empty_queue):
    """Test that a new queue is empty."""
    assert len(empty_queue) == 0
    assert not bool(empty_queue)


def test_push_single_item(empty_queue):
    """Test pushing a single item."""
    empty_queue.push("item1", 1)
    assert len(empty_queue) == 1
    assert bool(empty_queue)
    assert empty_queue.peek() == (1, "item1")


def test_push_multiple_items_different_priorities(empty_queue):
    """Test pushing multiple items with different priorities."""
    items = [("item3", 3), ("item1", 1), ("item2", 2)]
    for item, priority in items:
        empty_queue.push(item, priority)

    # Items should come out in priority order
    assert empty_queue.pop() == (1, "item1")
    assert empty_queue.pop() == (2, "item2")
    assert empty_queue.pop() == (3, "item3")


def test_push_multiple_items_same_priority(empty_queue):
    """Test FIFO behavior for items with equal priority."""
    items = [("first", 1), ("second", 1), ("third", 1)]
    for item, priority in items:
        empty_queue.push(item, priority)

    # Items should come out in insertion order
    assert empty_queue.pop() == (1, "first")
    assert empty_queue.pop() == (1, "second")
    assert empty_queue.pop() == (1, "third")


def test_mixed_priority_and_fifo(empty_queue):
    """Test mixed priority levels with some equal priorities."""
    empty_queue.push("medium1", 2)
    empty_queue.push("high", 1)
    empty_queue.push("medium2", 2)
    empty_queue.push("low", 3)

    # Check extraction order
    assert empty_queue.pop() == (1, "high")
    assert empty_queue.pop() == (2, "medium1")
    assert empty_queue.pop() == (2, "medium2")
    assert empty_queue.pop() == (3, "low")


def test_peek_behavior(empty_queue):
    """Test that peek returns items without removing them."""
    empty_queue.push("item1", 1)
    empty_queue.push("item2", 2)

    # Peek multiple times
    for _ in range(3):
        assert empty_queue.peek() == (1, "item1")
        assert len(empty_queue) == 2


def test_get_behavior(empty_queue):
    """Test the get method with both empty and non-empty queues."""
    # Test on empty queue
    assert empty_queue.get() is None

    # Test with items
    empty_queue.push("item1", 1)
    assert empty_queue.get() == (1, "item1")
    assert empty_queue.get() is None  # Queue should be empty again


def test_pop_empty_queue(empty_queue):
    """Test that pop raises IndexError on empty queue."""
    with pytest.raises(IndexError, match="pop from empty queue"):
        empty_queue.pop()


def test_peek_empty_queue(empty_queue):
    """Test that peek raises IndexError on empty queue."""
    with pytest.raises(IndexError, match="peek at empty queue"):
        empty_queue.peek()


def test_length_updates(empty_queue):
    """Test that length updates correctly with pushes and pops."""
    initial_len = len(empty_queue)
    assert initial_len == 0

    # Test length increases
    empty_queue.push("item1", 1)
    assert len(empty_queue) == 1
    empty_queue.push("item2", 2)
    assert len(empty_queue) == 2

    # Test length decreases
    empty_queue.pop()
    assert len(empty_queue) == 1
    empty_queue.pop()
    assert len(empty_queue) == 0


def test_bool_conversion(empty_queue):
    """Test boolean conversion in various states."""
    # Empty queue should be False
    assert not bool(empty_queue)

    # Queue with items should be True
    empty_queue.push("item", 1)
    assert bool(empty_queue)

    # Queue should be False again after removing item
    empty_queue.pop()
    assert not bool(empty_queue)


def test_large_number_of_items(empty_queue):
    """Test handling of a large number of items with mixed priorities."""
    # Add 1000 items with 10 different priority levels
    for i in range(1000):
        priority = i % 10
        empty_queue.push(f"item{i}", priority)

    # Verify FIFO order within same priority
    last_priority = -1
    last_index = -1
    for _ in range(1000):
        priority, item = empty_queue.pop()
        current_index = int(item[4:])  # Extract index from "itemX"

        if priority == last_priority:
            assert current_index > last_index, "FIFO order violated within same priority"
        else:
            assert priority > last_priority, "Priority order violated"

        last_priority = priority
        last_index = current_index


@pytest.mark.parametrize(
    "items",
    [
        [(1, "A"), (1, "B"), (1, "C")],  # Same priority
        [(3, "A"), (2, "B"), (1, "C")],  # Different priorities
        [(2, "A"), (1, "B"), (2, "C")],  # Mixed priorities
    ],
)
def test_queue_ordering_parametrized(empty_queue, items):
    """Parametrized test for different ordering scenarios."""
    for priority, item in items:
        empty_queue.push(item, priority)

    sorted_items = sorted(items, key=lambda x: (x[0], items.index(x)))
    for priority, item in sorted_items:
        assert empty_queue.pop() == (priority, item)


from haystack.tools import ComponentTool, Tool
from haystack.components.builders.prompt_builder import PromptBuilder


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


class TestDeepcopyWithFallback:
    def test_deepcopy_with_fallback_copyable(self, caplog):
        tool = Tool(
            name="weather",
            description="Get weather report",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=get_weather_report,
        )
        original = {"tools": tool}
        with caplog.at_level(logging.INFO):
            copy = deepcopy_with_fallback(original)
            assert "Deepcopy failed for object of type" not in caplog.text
        # This should be a true copy
        copy["tools"].name = "copied_tool"
        assert copy["tools"] != original["tools"]

    def test_deepcopy_with_fallback_not_copyable(self, caplog):
        problematic_tool = ComponentTool(
            name="problematic_tool", description="This is a problematic tool.", component=PromptBuilder("{{query}}")
        )
        original = {"tools": problematic_tool}
        with caplog.at_level(logging.INFO):
            copy = deepcopy_with_fallback(original)
            assert "Deepcopy failed for object of type 'ComponentTool'" in caplog.text
        # Not a true copy but a shallow copy
        copy["tools"].name = "copied_tool"
        assert copy["tools"] == original["tools"]

    def test_deepcopy_with_fallback_mixed_copyable_list(self, caplog):
        tool1 = Tool(
            name="tool1",
            description="Get weather report",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=get_weather_report,
        )
        tool2 = ComponentTool(
            name="problematic_tool", description="This is a problematic tool.", component=PromptBuilder("{{query}}")
        )
        original = {"tools": [tool1, tool2]}
        with caplog.at_level(logging.INFO):
            copy = deepcopy_with_fallback(original)
            assert "Deepcopy failed for object of type 'ComponentTool'" in caplog.text
        # First tool should be a true copy
        copy["tools"][0].name = "copied_tool1"
        assert copy["tools"][0] != original["tools"][0]
        # second should be a shallow copy
        copy["tools"][1].name = "copied_tool2"
        assert copy["tools"][1] == original["tools"][1]

    def test_deepcopy_with_fallback_mixed_copyable_tuple(self, caplog):
        tool1 = Tool(
            name="tool1",
            description="Get weather report",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=get_weather_report,
        )
        tool2 = ComponentTool(
            name="problematic_tool", description="This is a problematic tool.", component=PromptBuilder("{{query}}")
        )
        original = {"tools": (tool1, tool2)}
        with caplog.at_level(logging.INFO):
            copy = deepcopy_with_fallback(original)
            assert "Deepcopy failed for object of type 'ComponentTool'" in caplog.text
        # First tool should be a true copy
        copy["tools"][0].name = "copied_tool1"
        assert copy["tools"][0] != original["tools"][0]
        # second should be a shallow copy
        copy["tools"][1].name = "copied_tool2"
        assert copy["tools"][1] == original["tools"][1]
