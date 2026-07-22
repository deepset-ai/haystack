# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum

import pydantic
import pytest

from haystack.core.errors import DeserializationError, SerializationError
from haystack.dataclasses import ChatMessage, Document, GeneratedAnswer
from haystack.utils.base_serialization import _deserialize_value_with_schema, _serialize_value_with_schema


class PlainObject:
    """An arbitrary object without a ``to_dict`` method (only a `__dict__`)."""

    def __init__(self, value):
        self.value = value


class CustomModel(pydantic.BaseModel):
    id: int
    name: str


class CustomEnum(Enum):
    ONE = "one"
    TWO = "two"


def simple_calc_function(x: int) -> int:
    return x * 2


@pytest.mark.parametrize(
    "value,result",
    [
        # integer
        (1, {"serialization_schema": {"type": "integer"}, "serialized_data": 1}),
        # float
        (1.5, {"serialization_schema": {"type": "number"}, "serialized_data": 1.5}),
        # string
        ("test", {"serialization_schema": {"type": "string"}, "serialized_data": "test"}),
        # boolean
        (True, {"serialization_schema": {"type": "boolean"}, "serialized_data": True}),
        (False, {"serialization_schema": {"type": "boolean"}, "serialized_data": False}),
        # None
        (None, {"serialization_schema": {"type": "null"}, "serialized_data": None}),
    ],
)
def test_serialize_and_deserialize_primitive_types(value, result):
    assert _serialize_value_with_schema(value) == result
    assert _deserialize_value_with_schema(result) == value


@pytest.mark.parametrize(
    "value,result",
    [
        # empty dict
        ({}, {"serialization_schema": {"type": "object", "properties": {}}, "serialized_data": {}}),
        # empty list
        ([], {"serialization_schema": {"type": "array", "items": {}}, "serialized_data": []}),
        # empty tuple
        (
            (),
            {
                "serialization_schema": {"type": "array", "items": {}, "minItems": 0, "maxItems": 0},
                "serialized_data": [],
            },
        ),
        # empty set
        (set(), {"serialization_schema": {"type": "array", "items": {}, "uniqueItems": True}, "serialized_data": []}),
        # nested empty structures
        (
            {"empty_list": [], "empty_dict": {}, "nested_empty": {"empty": []}},
            {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "empty_list": {"type": "array", "items": {}},
                        "empty_dict": {"type": "object", "properties": {}},
                        "nested_empty": {"type": "object", "properties": {"empty": {"type": "array", "items": {}}}},
                    },
                },
                "serialized_data": {"empty_list": [], "empty_dict": {}, "nested_empty": {"empty": []}},
            },
        ),
    ],
)
def test_serializing_and_deserializing_empty_structures(value, result):
    assert _serialize_value_with_schema(value) == result
    assert _deserialize_value_with_schema(result) == value


@pytest.mark.parametrize(
    "value,result",
    [
        # list
        (
            [1, 2, 3],
            {"serialization_schema": {"type": "array", "items": {"type": "integer"}}, "serialized_data": [1, 2, 3]},
        ),
        # set
        (
            {1, 2, 3},
            {
                "serialization_schema": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                "serialized_data": [1, 2, 3],
            },
        ),
        # frozenset
        (
            frozenset({1, 2, 3}),
            {
                "serialization_schema": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "uniqueItems": True,
                    "frozen": True,
                },
                "serialized_data": [1, 2, 3],
            },
        ),
        # tuple
        (
            (1, 2, 3),
            {
                "serialization_schema": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                "serialized_data": [1, 2, 3],
            },
        ),
        # nested list
        (
            [[1, 2], [3, 4]],
            {
                "serialization_schema": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                "serialized_data": [[1, 2], [3, 4]],
            },
        ),
        # list of set
        (
            [{1, 2}, {3, 4}],
            {
                "serialization_schema": {
                    "items": {"items": {"type": "integer"}, "type": "array", "uniqueItems": True},
                    "type": "array",
                },
                "serialized_data": [[1, 2], [3, 4]],
            },
        ),
        # nested tuple
        (
            ((1, 2), (3, 4), (5, 6)),
            {
                "serialization_schema": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "serialized_data": [[1, 2], [3, 4], [5, 6]],
            },
        ),
        # nested list of GeneratedAnswer
        (
            [
                [
                    GeneratedAnswer(
                        data="Paris",
                        query="What is the capital of France?",
                        documents=[Document(content="Paris is the capital of France", id="1")],
                        meta={"page": 1},
                    )
                ],
                [
                    GeneratedAnswer(
                        data="Berlin",
                        query="What is the capital of Germany?",
                        documents=[Document(content="Berlin is the capital of Germany", id="2")],
                        meta={"page": 1},
                    )
                ],
            ],
            {
                "serialization_schema": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "haystack.dataclasses.answer.GeneratedAnswer"}},
                },
                "serialized_data": [
                    [
                        {
                            "data": "Paris",
                            "query": "What is the capital of France?",
                            "documents": [
                                {
                                    "id": "1",
                                    "content": "Paris is the capital of France",
                                    "blob": None,
                                    "meta": {},
                                    "score": None,
                                    "embedding": None,
                                    "sparse_embedding": None,
                                }
                            ],
                            "meta": {"page": 1},
                        }
                    ],
                    [
                        {
                            "data": "Berlin",
                            "query": "What is the capital of Germany?",
                            "documents": [
                                {
                                    "id": "2",
                                    "content": "Berlin is the capital of Germany",
                                    "blob": None,
                                    "meta": {},
                                    "score": None,
                                    "embedding": None,
                                    "sparse_embedding": None,
                                }
                            ],
                            "meta": {"page": 1},
                        }
                    ],
                ],
            },
        ),
    ],
)
def test_serialize_and_deserialize_sequence_types(value, result):
    assert _serialize_value_with_schema(value) == result
    deserialized = _deserialize_value_with_schema(result)
    assert deserialized == value
    # `frozenset({...}) == set({...})` is True, so check the exact type to catch container regressions.
    assert type(deserialized) is type(value)


@pytest.mark.parametrize(
    "value,result",
    [
        pytest.param(
            {"key1": {"nested1": "value1", "nested2": {"deep": "value2"}}},
            {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "key1": {
                            "type": "object",
                            "properties": {
                                "nested1": {"type": "string"},
                                "nested2": {"type": "object", "properties": {"deep": {"type": "string"}}},
                            },
                        }
                    },
                },
                "serialized_data": {"key1": {"nested1": "value1", "nested2": {"deep": "value2"}}},
            },
            id="nested-dicts",
        ),
        pytest.param(
            simple_calc_function,
            {
                "serialization_schema": {"type": "typing.Callable"},
                "serialized_data": "test_base_serialization.simple_calc_function",
            },
            id="callable",
        ),
        pytest.param(
            CustomEnum.ONE,
            {"serialization_schema": {"type": "test_base_serialization.CustomEnum"}, "serialized_data": "ONE"},
            id="enum",
        ),
        pytest.param(
            CustomModel(id=1, name="Test"),
            {
                "serialization_schema": {"type": "test_base_serialization.CustomModel"},
                "serialized_data": {"id": 1, "name": "Test"},
            },
            id="pydantic-model",
        ),
    ],
)
def test_serialize_and_deserialize_complex_types(value, result):
    assert _serialize_value_with_schema(value) == result
    assert _deserialize_value_with_schema(result) == value


def test_serialize_and_deserialize_value_with_schema_with_various_types():
    data = {
        "numbers": 1,
        "key_name": None,
        "messages": [ChatMessage.from_user(text="Hello, world!"), ChatMessage.from_assistant(text="Hello, world!")],
        "user_id": "123",
        "dict_of_lists": {"numbers": [1, 2, 3]},
        "documents": [Document(content="Hello, world!", id="1")],
        "list_of_dicts": [{"numbers": [1, 2, 3]}],
        "answers": [
            GeneratedAnswer(
                data="Paris",
                query="What is the capital of France?",
                documents=[Document(content="Paris is the capital of France", id="2")],
                meta={"page": 1},
            )
        ],
    }
    expected = {
        "serialization_schema": {
            "type": "object",
            "properties": {
                "numbers": {"type": "integer"},
                "key_name": {"type": "null"},
                "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
                "user_id": {"type": "string"},
                "dict_of_lists": {
                    "type": "object",
                    "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}},
                },
                "documents": {"type": "array", "items": {"type": "haystack.dataclasses.document.Document"}},
                "list_of_dicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}},
                    },
                },
                "answers": {"type": "array", "items": {"type": "haystack.dataclasses.answer.GeneratedAnswer"}},
            },
        },
        "serialized_data": {
            "numbers": 1,
            "key_name": None,
            "messages": [
                {"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]},
                {"role": "assistant", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]},
            ],
            "user_id": "123",
            "dict_of_lists": {"numbers": [1, 2, 3]},
            "documents": [
                {
                    "id": "1",
                    "content": "Hello, world!",
                    "blob": None,
                    "score": None,
                    "embedding": None,
                    "sparse_embedding": None,
                }
            ],
            "list_of_dicts": [{"numbers": [1, 2, 3]}],
            "answers": [
                {
                    "data": "Paris",
                    "query": "What is the capital of France?",
                    "documents": [
                        {
                            "id": "2",
                            "content": "Paris is the capital of France",
                            "blob": None,
                            "meta": {},
                            "score": None,
                            "embedding": None,
                            "sparse_embedding": None,
                        }
                    ],
                    "meta": {"page": 1},
                }
            ],
        },
    }
    assert _serialize_value_with_schema(data) == expected
    assert _deserialize_value_with_schema(expected) == data


class TestErrorHandling:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(datetime(2024, 1, 1), id="datetime"),
            pytest.param(b"some bytes", id="bytes"),
            pytest.param(3 + 4j, id="complex"),
            pytest.param(PlainObject(1), id="object-without-to_dict"),
        ],
    )
    def test_serialize_unsupported_type_raises(self, value):
        with pytest.raises(SerializationError, match="Cannot serialize value of type"):
            _serialize_value_with_schema(value)

    def test_serialize_unsupported_nested_value_raises(self):
        # An unsupported value nested inside a supported container must not be silently passed through.
        with pytest.raises(SerializationError, match="Cannot serialize value of type"):
            _serialize_value_with_schema({"good": 1, "bad": datetime(2024, 1, 1)})

    def test_deserialize_value_with_wrong_value(self):
        with pytest.raises(DeserializationError, match="Value 'NOT_VALID' is not a valid member of Enum"):
            _deserialize_value_with_schema(
                {"serialization_schema": {"type": "test_base_serialization.CustomEnum"}, "serialized_data": "NOT_VALID"}
            )

    def test_deserialize_value_with_schema_class_not_importable(self):
        with pytest.raises(
            DeserializationError, match="Class 'test_base_serialization.NonExistentClass' not correctly imported"
        ):
            _deserialize_value_with_schema(
                {"serialization_schema": {"type": "test_base_serialization.NonExistentClass"}, "serialized_data": {}}
            )

    def test_deserialize_value_with_schema_class_name_without_module(self):
        with pytest.raises(DeserializationError, match="Class 'NonExistentClass' not correctly imported"):
            _deserialize_value_with_schema(
                {"serialization_schema": {"type": "NonExistentClass"}, "serialized_data": {}}
            )

    def test_deserialize_pydantic_model_with_invalid_data(self):
        with pytest.raises(
            DeserializationError,
            match="Failed to deserialize data '{'id': 'not_an_integer', 'name': 'Test'}' into "
            "Pydantic model 'test_base_serialization.CustomModel'",
        ):
            _deserialize_value_with_schema(
                {
                    "serialization_schema": {"type": "test_base_serialization.CustomModel"},
                    "serialized_data": {"id": "not_an_integer", "name": "Test"},
                }
            )
