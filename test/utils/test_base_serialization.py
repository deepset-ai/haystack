# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.core.errors import DeserializationError, SerializationError
from haystack.dataclasses import ChatMessage, Document, GeneratedAnswer
from haystack.utils.base_serialization import (
    _deserialize_value_with_schema,
    _serialize_value_with_schema,
    deserialize_class_instance,
    serialize_class_instance,
)


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


def test_serialize_value_primitive_types():
    numbers = 1
    string = "test"
    _bool = True
    none = None
    result = _serialize_value_with_schema(numbers)
    assert result == {"serialization_schema": {"type": "integer"}, "serialized_data": 1}
    result = _serialize_value_with_schema(string)
    assert result == {"serialization_schema": {"type": "string"}, "serialized_data": "test"}
    result = _serialize_value_with_schema(_bool)
    assert result == {"serialization_schema": {"type": "boolean"}, "serialized_data": True}
    result = _serialize_value_with_schema(none)
    assert result == {"serialization_schema": {"type": "null"}, "serialized_data": None}


def test_deserialize_value_primitive_types():
    result = _deserialize_value_with_schema({"serialization_schema": {"type": "integer"}, "serialized_data": 1})
    assert result == 1
    result = _deserialize_value_with_schema({"serialization_schema": {"type": "string"}, "serialized_data": "test"})
    assert result == "test"
    result = _deserialize_value_with_schema({"serialization_schema": {"type": "boolean"}, "serialized_data": True})
    assert result == True
    result = _deserialize_value_with_schema({"serialization_schema": {"type": "null"}, "serialized_data": None})
    assert result == None


def test_serialize_value_with_sequences():
    sequences = [1, 2, 3]
    set_sequences = {1, 2, 3}
    tuple_sequences = (1, 2, 3)
    result = _serialize_value_with_schema(sequences)
    assert result == {
        "serialization_schema": {"type": "array", "items": {"type": "integer"}},
        "serialized_data": [1, 2, 3],
    }
    result = _serialize_value_with_schema(set_sequences)
    assert result == {
        "serialization_schema": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        "serialized_data": [1, 2, 3],
    }
    result = _serialize_value_with_schema(tuple_sequences)
    assert result == {
        "serialization_schema": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
        "serialized_data": [1, 2, 3],
    }


def test_deserialize_value_with_sequences():
    sequences = [1, 2, 3]
    set_sequences = {1, 2, 3}
    tuple_sequences = (1, 2, 3)
    result = _deserialize_value_with_schema(
        {"serialization_schema": {"type": "array", "items": {"type": "integer"}}, "serialized_data": [1, 2, 3]}
    )
    assert result == sequences
    result = _deserialize_value_with_schema(
        {
            "serialization_schema": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
            "serialized_data": [1, 2, 3],
        }
    )
    assert result == set_sequences
    result = _deserialize_value_with_schema(
        {
            "serialization_schema": {
                "type": "array",
                "items": {"type": "integer"},
                "collection_type": "tuple",
                "minItems": 3,
                "maxItems": 3,
            },
            "serialized_data": [1, 2, 3],
        }
    )
    assert result == tuple_sequences


def test_serializing_and_deserializing_nested_lists():
    nested_lists = [[1, 2], [3, 4]]

    serialized_nested_lists = _serialize_value_with_schema(nested_lists)
    assert serialized_nested_lists == {
        "serialization_schema": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "serialized_data": [[1, 2], [3, 4]],
    }

    deserialized_nested_lists = _deserialize_value_with_schema(serialized_nested_lists)
    assert deserialized_nested_lists == nested_lists


def test_serializing_and_deserializing_nested_answer_lists():
    """Test that _deserialize_value_with_schema handles nested lists"""

    nested_answers_list = [
        [
            GeneratedAnswer(
                data="Paris",
                query="What is the capital of France?",
                documents=[Document(content="Paris is the capital of France")],
                meta={"page": 1},
            )
        ],
        [
            GeneratedAnswer(
                data="Berlin",
                query="What is the capital of Germany?",
                documents=[Document(content="Berlin is the capital of Germany")],
                meta={"page": 1},
            )
        ],
    ]
    serialized_nested_answers_list = _serialize_value_with_schema(nested_answers_list)
    assert serialized_nested_answers_list == {
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
                                "id": "413dccdf51a54cca75b7ed2eddac04e6e58560bd2f0caf4106a3efc023fe3651",
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
                                "id": "c7b5b839963fcbf9b394b24c883731e840c3170ace33afb7af87a2de8a257f6f",
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
    }

    deserialized_nested_answers_list = _deserialize_value_with_schema(serialized_nested_answers_list)
    assert deserialized_nested_answers_list == nested_answers_list


def test_serializing_and_deserializing_nested_dicts():
    data = {"key1": {"nested1": "value1", "nested2": {"deep": "value2"}}}
    serialized_nested_dicts = _serialize_value_with_schema(data)
    assert serialized_nested_dicts == {
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

    deserialized_nested_dicts = _deserialize_value_with_schema(serialized_nested_dicts)
    assert deserialized_nested_dicts == data


def test_serializing_and_deserializing_nested_sets():
    nested_sets = [{1, 2}, {3, 4}]

    result = _serialize_value_with_schema(nested_sets)
    assert result == {
        "serialization_schema": {
            "items": {"items": {"type": "integer"}, "type": "array", "uniqueItems": True},
            "type": "array",
        },
        "serialized_data": [[1, 2], [3, 4]],
    }

    result = _deserialize_value_with_schema(
        {
            "serialization_schema": {
                "items": {"items": {"type": "integer"}, "type": "array", "uniqueItems": True},
                "type": "array",
            },
            "serialized_data": [[1, 2], [3, 4]],
        }
    )
    assert result == nested_sets


def test_serializing_and_deserializing_empty_structures():
    """Test that _deserialize_value_with_schema handles empty structures"""
    data = {"empty_list": [], "empty_dict": {}, "nested_empty": {"empty": []}}
    serialized_data = _serialize_value_with_schema(data)
    result = _deserialize_value_with_schema(serialized_data)

    assert result == data


def test_serialize_value_with_schema():
    data = {
        "numbers": 1,
        "messages": [ChatMessage.from_user(text="Hello, world!"), ChatMessage.from_assistant(text="Hello, world!")],
        "user_id": "123",
        "dict_of_lists": {"numbers": [1, 2, 3]},
        "documents": [Document(content="Hello, world!")],
        "list_of_dicts": [{"numbers": [1, 2, 3]}],
        "answers": [
            GeneratedAnswer(
                data="Paris",
                query="What is the capital of France?",
                documents=[Document(content="Paris is the capital of France")],
                meta={"page": 1},
            )
        ],
    }
    result = _serialize_value_with_schema(data)
    assert result == {
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
                    "id": "e0f8c9e42f5535600aee6c5224bf4478b73bcf0a1bcba6f357bf162e88ff985d",
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
                                "id": "413dccdf51a54cca75b7ed2eddac04e6e58560bd2f0caf4106a3efc023fe3651",
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


def test_deserialize_value_with_schema():
    serialized__data = {
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
                "list_of_dicts": {"type": "array", "items": {"type": "string"}},
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
                    "id": "e0f8c9e42f5535600aee6c5224bf4478b73bcf0a1bcba6f357bf162e88ff985d",
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
                                "id": "413dccdf51a54cca75b7ed2eddac04e6e58560bd2f0caf4106a3efc023fe3651",
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

    result = _deserialize_value_with_schema(serialized__data)
    assert result["numbers"] == 1
    assert isinstance(result["messages"][0], ChatMessage)
    assert result["messages"][0].text == "Hello, world!"
    assert result["user_id"] == "123"
    assert result["dict_of_lists"] == {"numbers": [1, 2, 3]}
    assert isinstance(result["documents"][0], Document)
    assert result["documents"][0].content == "Hello, world!"
    assert isinstance(result["answers"][0], GeneratedAnswer)


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
