# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import pytest

from haystack import Document
from haystack.tracing import utils
from haystack.tracing.utils import _serializable_value


class NonSerializableClass:
    def __str__(self) -> str:
        return "NonSerializableClass"


class ClassWithToDict:
    def __init__(self, value: str):
        self.value = value

    def to_dict(self) -> dict:
        return {"value": self.value}


class ClassWithBothMethods:
    """Class with both to_dict and _to_trace_dict methods."""

    def __init__(self, data: bytes):
        self.data = data

    def to_dict(self) -> dict:
        return {"data": list(self.data)}

    def _to_trace_dict(self) -> dict:
        return {"data": f"Binary ({len(self.data)} bytes)"}


class TestSerializableValue:
    @pytest.mark.parametrize("value", [1, 1.0, True, False, "string", None])
    def test_primitive_types(self, value: Any) -> None:
        assert _serializable_value(value) == value

    def test_list_serialized_recursively(self) -> None:
        result = _serializable_value([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_dict_serialized_recursively(self) -> None:
        result = _serializable_value({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_nested_list_and_dict(self) -> None:
        value = {"items": [1, 2, {"nested": "value"}]}
        result = _serializable_value(value)
        assert result == {"items": [1, 2, {"nested": "value"}]}

    def test_object_with_to_dict(self) -> None:
        obj = ClassWithToDict("test")
        result = _serializable_value(obj)
        assert result == {"value": "test"}

    def test_object_with_to_trace_dict_placeholders(self) -> None:
        obj = ClassWithBothMethods(b"hello")
        result = _serializable_value(obj, use_placeholders=True)
        assert result == {"data": "Binary (5 bytes)"}

    def test_object_with_to_trace_dict_no_placeholders(self) -> None:
        obj = ClassWithBothMethods(b"hello")
        result = _serializable_value(obj, use_placeholders=False)
        assert result == {"data": [104, 101, 108, 108, 111]}

    def test_list_of_objects_with_to_dict(self) -> None:
        objs = [ClassWithToDict("a"), ClassWithToDict("b")]
        result = _serializable_value(objs)
        assert result == [{"value": "a"}, {"value": "b"}]

    def test_dict_with_object_values(self) -> None:
        value = {"obj": ClassWithToDict("test")}
        result = _serializable_value(value)
        assert result == {"obj": {"value": "test"}}

    def test_object_without_serialization_methods(self) -> None:
        obj = NonSerializableClass()
        result = _serializable_value(obj)
        assert result is obj


class TestTypeCoercion:
    @pytest.mark.parametrize(
        "raw_value,expected_tag_value",
        [
            (1, 1),
            (1.0, 1.0),
            (True, True),
            (None, ""),
            ("string", "string"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({"key": "value"}, '{"key": "value"}'),
            (NonSerializableClass(), "NonSerializableClass"),
            (
                Document(id="1", content="text"),
                '{"id": "1", "content": "text", "blob": null, "score": null, "embedding": null, '
                '"sparse_embedding": null}',
            ),
            (
                [Document(id="1", content="text")],
                '[{"id": "1", "content": "text", "blob": null, "score": null, "embedding": null, '
                '"sparse_embedding": null}]',
            ),
            (
                {"key": Document(id="1", content="text")},
                '{"key": {"id": "1", "content": "text", "blob": null, "score": null, "embedding": null, '
                '"sparse_embedding": null}}',
            ),
        ],
    )
    def test_type_coercion(self, raw_value: Any, expected_tag_value: Union[bool, str, int, float]) -> None:
        coerced_value = utils.coerce_tag_value(raw_value)

        assert coerced_value == expected_tag_value
