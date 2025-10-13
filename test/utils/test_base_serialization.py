# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest

from haystack.core.errors import DeserializationError, SerializationError
from haystack.dataclasses import ChatMessage, Document, GeneratedAnswer
from haystack.utils.base_serialization import (
    _deserialize_value_with_schema,
    _serialize_value_with_schema,
    deserialize_class_instance,
    serialize_class_instance,
)


class CustomEnum(Enum):
    ONE = "one"
    TWO = "two"


class CustomClass:
    def to_dict(self):
        return {"key": "value", "more": False}

    @classmethod
    def from_dict(cls, data):
        assert data == {"key": "value", "more": False}
        return cls()


class CustomClassNoToDict:
    @classmethod
    def from_dict(cls, data):
        assert data == {"key": "value", "more": False}
        return cls()


class CustomClassNoFromDict:
    def to_dict(self):
        return {"key": "value", "more": False}


def simple_calc_function(x: int) -> int:
    return x * 2


def test_serialize_class_instance():
    result = serialize_class_instance(CustomClass())
    assert result == {"data": {"key": "value", "more": False}, "type": "test_base_serialization.CustomClass"}


def test_serialize_class_instance_missing_method():
    with pytest.raises(SerializationError, match="does not have a 'to_dict' method"):
        serialize_class_instance(CustomClassNoToDict())


def test_deserialize_class_instance():
    data = {"data": {"key": "value", "more": False}, "type": "test_base_serialization.CustomClass"}

    result = deserialize_class_instance(data)
    assert isinstance(result, CustomClass)


def test_deserialize_class_instance_invalid_data():
    with pytest.raises(DeserializationError, match="Missing 'type'"):
        deserialize_class_instance({})

    with pytest.raises(DeserializationError, match="Missing 'data'"):
        deserialize_class_instance({"type": "test_base_serialization.CustomClass"})

    with pytest.raises(
        DeserializationError, match="Class 'test_base_serialization.CustomClass1' not correctly imported"
    ):
        deserialize_class_instance({"type": "test_base_serialization.CustomClass1", "data": {}})

    with pytest.raises(
        DeserializationError,
        match="Class 'test_base_serialization.CustomClassNoFromDict' does not have a 'from_dict' method",
    ):
        deserialize_class_instance({"type": "test_base_serialization.CustomClassNoFromDict", "data": {}})


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
                            "type": "haystack.dataclasses.answer.GeneratedAnswer",
                            "init_parameters": {
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
                            },
                        }
                    ],
                    [
                        {
                            "type": "haystack.dataclasses.answer.GeneratedAnswer",
                            "init_parameters": {
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
                            },
                        }
                    ],
                ],
            },
        ),
    ],
)
def test_serialize_and_deserialize_sequence_types(value, result):
    assert _serialize_value_with_schema(value) == result
    assert _deserialize_value_with_schema(result) == value


def test_serialize_and_deserialize_nested_dicts():
    data = {"key1": {"nested1": "value1", "nested2": {"deep": "value2"}}}
    expected = {
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
    }
    assert _serialize_value_with_schema(data) == expected
    assert _deserialize_value_with_schema(expected) == data


def test_serialize_and_deserialize_value_with_schema_with_various_types():
    data = {
        "numbers": 1,
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
                    "type": "haystack.dataclasses.answer.GeneratedAnswer",
                    "init_parameters": {
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
                    },
                }
            ],
        },
    }
    assert _serialize_value_with_schema(data) == expected
    assert _deserialize_value_with_schema(expected) == data


def test_serializing_and_deserializing_custom_class_type():
    custom_type = CustomClass()
    data = {"numbers": 1, "custom_type": custom_type}
    serialized_data = _serialize_value_with_schema(data)
    assert serialized_data == {
        "serialization_schema": {
            "properties": {
                "custom_type": {"type": "test_base_serialization.CustomClass"},
                "numbers": {"type": "integer"},
            },
            "type": "object",
        },
        "serialized_data": {"numbers": 1, "custom_type": {"key": "value", "more": False}},
    }

    deserialized_data = _deserialize_value_with_schema(serialized_data)
    assert deserialized_data["numbers"] == 1
    assert isinstance(deserialized_data["custom_type"], CustomClass)


def test_serialize_and_deserialize_value_with_callable():
    expected = {
        "serialization_schema": {"type": "typing.Callable"},
        "serialized_data": "test_base_serialization.simple_calc_function",
    }
    assert _serialize_value_with_schema(simple_calc_function) == expected
    assert _deserialize_value_with_schema(expected) == simple_calc_function


# def test_serialize_and_deserialize_value_with_enum():
#     data = CustomEnum.ONE
#     expected = {"serialization_schema": {"type": "test_base_serialization.CustomEnum"}, "serialized_data": "ONE"}
#     assert _serialize_value_with_schema(data) == expected
#     assert _deserialize_value_with_schema(expected) == data
