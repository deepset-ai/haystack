# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copyreg
import logging
import warnings
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

import pytest

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.core.pipeline.utils import (
    FIFOPriorityQueue,
    _deepcopy_with_exceptions,
    args_deprecated,
    parse_connect_string,
)
from haystack.dataclasses import ByteStream, Document, SparseEmbedding
from haystack.tools import ComponentTool


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


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


class Copyable:
    def __init__(self, name="copyable"):
        self.name = name


class NotCopyable:
    def __init__(self, name="not_copyable"):
        self.name = name

    def __deepcopy__(self, memo):
        raise TypeError("This object cannot be deepcopied.")


class TestDeepcopyWithFallback:
    def test_deepcopy_with_fallback_copyable(self, caplog):
        original = {"class": Copyable()}
        with caplog.at_level(logging.INFO):
            copy = _deepcopy_with_exceptions(original)
            assert "Deepcopy failed for object of type" not in caplog.text
        assert copy["class"] is not original["class"]

    def test_deepcopy_with_fallback_not_copyable_error(self, caplog):
        original = {"class": NotCopyable()}
        with caplog.at_level(logging.INFO):
            copy = _deepcopy_with_exceptions(original)
            assert "Deepcopy failed for object of type 'NotCopyable'" in caplog.text
        assert copy["class"] is original["class"]

    def test_deepcopy_with_fallback_mixed_copyable_list(self, caplog):
        obj1 = Copyable()
        obj2 = NotCopyable()
        original = {"objects": [obj1, obj2]}
        with caplog.at_level(logging.INFO):
            copy = _deepcopy_with_exceptions(original)
            assert "Deepcopy failed for object of type 'NotCopyable'" in caplog.text
        assert copy["objects"][0] is not original["objects"][0]
        assert copy["objects"][1] is original["objects"][1]

    def test_deepcopy_with_fallback_mixed_copyable_tuple(self, caplog):
        obj1 = Copyable()
        obj2 = NotCopyable()
        original = {"objects": (obj1, obj2)}
        with caplog.at_level(logging.INFO):
            copy = _deepcopy_with_exceptions(original)
            assert "Deepcopy failed for object of type 'NotCopyable'" in caplog.text
        assert copy["objects"][0] is not original["objects"][0]
        assert copy["objects"][1] is original["objects"][1]

    def test_deepcopy_with_fallback_tool(self):
        tool = ComponentTool(
            name="problematic_tool", description="This is a problematic tool.", component=PromptBuilder("{{query}}")
        )
        original = {"tools": tool}
        copy = _deepcopy_with_exceptions(original)
        assert copy["tools"] is original["tools"]

    def test_deepcopy_with_fallback_component(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        comp = OpenAIChatGenerator()
        original = {"component": comp}
        res = _deepcopy_with_exceptions(original)
        assert res["component"] is original["component"]

    def test_deepcopy_with_fallback_namedtuple(self):
        Point = namedtuple("Point", ["x", "y"])
        inner = Copyable()
        original = {"point": Point(inner, 2)}
        copy = _deepcopy_with_exceptions(original)
        # A namedtuple must be reconstructed as its own type. Its __new__ takes
        # positional fields, so the plain ``type(obj)(<generator>)`` path used for
        # lists/tuples/sets would raise a TypeError instead of copying it.
        assert isinstance(copy["point"], Point)
        assert copy["point"].y == 2
        # Its contents are deep-copied, matching how plain tuples are handled.
        assert copy["point"].x is not original["point"].x

    @pytest.mark.parametrize("value", [None, True, 3, 2.5, 1 + 2j, "text", b"bytes"])
    def test_deepcopy_with_fallback_matches_deepcopy_for_immutable_builtins(self, value):
        original = {"values": [value], "value": value}
        copy = _deepcopy_with_exceptions(original)
        assert copy["values"] is not original["values"]
        assert copy == original
        assert copy["value"] is deepcopy(value)
        assert copy["values"][0] is deepcopy(value)

    def test_deepcopy_with_fallback_copies_subclass_of_immutable_builtin(self):
        class TaggedStr(str):
            __slots__ = ()

            def __deepcopy__(self, memo):
                return TaggedStr(f"{self} (copied)")

        copy = _deepcopy_with_exceptions(TaggedStr("text"))
        assert type(copy) is TaggedStr
        assert copy == "text (copied)"

    @pytest.mark.parametrize(
        "document",
        [
            Document(content="text"),
            Document(content="text", embedding=[0.1, 0.2, 0.3], score=0.5, meta={"page": 1}),
            Document(content="text", embedding=[], meta={"nested": {"tags": ["a", "b"]}, "items": [{"k": 1}]}),
            Document(
                blob=ByteStream(data=b"data"), sparse_embedding=SparseEmbedding(indices=[0, 2], values=[0.5, 1.0])
            ),
            Document(content="text", embedding=[1, 2.0], meta={"mixed": [1, "two", [3]]}),
        ],
    )
    def test_deepcopy_with_fallback_document_matches_deepcopy(self, document):
        copy = _deepcopy_with_exceptions(document)
        expected = deepcopy(document)
        assert type(copy) is Document
        assert copy == expected
        assert copy.__dict__ == expected.__dict__
        for name, value in document.__dict__.items():
            if type(value) not in (type(None), int, float, str):
                assert copy.__dict__[name] is not value

    def test_deepcopy_with_fallback_document_copies_nested_meta(self):
        document = Document(content="text", meta={"nested": {"tags": ["a"]}})
        copy = _deepcopy_with_exceptions(document)
        copy.meta["nested"]["tags"].append("b")
        assert document.meta == {"nested": {"tags": ["a"]}}

    @pytest.mark.parametrize("alias_in_meta_first", [True, False])
    def test_deepcopy_with_fallback_document_keeps_aliasing_like_deepcopy(self, alias_in_meta_first):
        embedding = [0.1, 0.2]
        if alias_in_meta_first:
            document = Document(content="text", meta={"same": embedding}, embedding=embedding)
        else:
            document = Document(content="text", embedding=embedding)
            document.__dict__["extra"] = embedding
        copy = _deepcopy_with_exceptions(document)
        expected = deepcopy(document)
        aliased_name = "meta" if alias_in_meta_first else "extra"
        copied_alias = copy.meta["same"] if alias_in_meta_first else copy.__dict__[aliased_name]
        expected_alias = expected.meta["same"] if alias_in_meta_first else expected.__dict__[aliased_name]
        assert (copied_alias is copy.embedding) == (expected_alias is expected.embedding) is True
        assert copy.embedding is not embedding

    def test_deepcopy_with_fallback_document_keeps_extra_attributes(self):
        document = Document(content="text")
        document.__dict__["extra"] = [{"k": 1}]
        document.__dict__["flags"] = [1, None, "x"]
        copy = _deepcopy_with_exceptions(document)
        assert copy.__dict__["extra"] == [{"k": 1}]
        assert copy.__dict__["extra"][0] is not document.__dict__["extra"][0]
        assert copy.__dict__["flags"] == [1, None, "x"]
        assert copy.__dict__["flags"] is not document.__dict__["flags"]

    def test_deepcopy_with_fallback_document_subclass_uses_deepcopy(self):
        @dataclass
        class CustomDocument(Document):
            def __deepcopy__(self, memo):
                return "custom copy"

        assert _deepcopy_with_exceptions(CustomDocument(content="text")) == "custom copy"

    def test_deepcopy_with_fallback_document_matching_component_protocol_is_not_copied(self):
        document = Document(content="text")
        document.__dict__["run"] = dict
        assert _deepcopy_with_exceptions(document) is document

    @pytest.mark.parametrize(
        "name, hook",
        [("__deepcopy__", lambda memo: "custom copy"), ("__reduce_ex__", lambda protocol: (str, ("custom copy",)))],
    )
    def test_deepcopy_with_fallback_document_instance_copy_hook_is_used(self, name, hook):
        document = Document(content="text")
        document.__dict__[name] = hook
        assert _deepcopy_with_exceptions(document) == "custom copy"

    def test_deepcopy_with_fallback_document_instance_copy_hook_raising_returns_original(self, caplog):
        def failing_hook(memo):
            raise RuntimeError("copy failed")

        document = Document(content="text")
        document.__dict__["__deepcopy__"] = failing_hook
        with caplog.at_level(logging.INFO):
            assert _deepcopy_with_exceptions(document) is document
            assert "Deepcopy failed for object of type 'Document'" in caplog.text

    def test_deepcopy_with_fallback_document_copyreg_reducer_is_used(self, monkeypatch):
        monkeypatch.setitem(copyreg.dispatch_table, Document, lambda document: (str, ("custom copy",)))
        assert _deepcopy_with_exceptions(Document(content="text")) == "custom copy"

    def test_document_uses_default_copy_protocol(self):
        # `_deepcopy_document` mirrors how `deepcopy` copies a plain dataclass. If `Document` customizes copying,
        # the fast path must be updated or removed, otherwise it would silently bypass the customization.
        assert "__deepcopy__" not in vars(Document)
        assert Document.__reduce_ex__ is object.__reduce_ex__
        assert Document.__reduce__ is object.__reduce__
        assert Document.__getstate__ is object.__getstate__
        assert not hasattr(Document, "__slots__")

    def test_deepcopy_with_fallback_document_not_copyable(self, caplog):
        document = Document(content="text", meta={"obj": NotCopyable()})
        with caplog.at_level(logging.INFO):
            copy = _deepcopy_with_exceptions(document)
            assert "Deepcopy failed for object of type 'Document'" in caplog.text
        assert copy is document


class TestArgsDeprecated:
    @pytest.fixture
    def sample_function(self):
        @args_deprecated
        def sample_func(param1: str = "default1", param2: int = 42) -> str:
            return f"{param1}-{param2}"

        return sample_func

    def test_warning_with_positional_args(self, sample_function):
        # using positional arguments only
        with warnings.catch_warnings(record=True) as w:
            result = sample_function("test", 123)
            assert result == "test-123"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert (
                "Warning: In an upcoming release, this method will require keyword arguments for all parameters"
                in str(w[0].message)
            )

    def test_warning_with_mixed_args(self, sample_function):
        # mixing positional and keyword arguments
        with warnings.catch_warnings(record=True) as w:
            result = sample_function("test", 123)
            assert result == "test-123"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert (
                "Warning: In an upcoming release, this method will require keyword arguments for all parameters"
                in str(w[0].message)
            )

    def test_no_warning_with_default_args(self, sample_function):
        # using default arguments
        with warnings.catch_warnings(record=True) as w:
            result = sample_function()
            assert result == "default1-42"
            assert len(w) == 0

    def test_no_warning_with_keyword_args(self, sample_function):
        # using keyword arguments
        with warnings.catch_warnings(record=True) as w:
            result = sample_function(param1="test", param2=123)
            assert result == "test-123"
            assert len(w) == 0
