# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.core.errors import DeserializationError, SerializationError
import pytest

from haystack.utils.base_serialization import (
    serialize_class_instance,
    deserialize_class_instance,
    serialize_value_with_schema,
    deserialize_value_with_schema,
)
from haystack.dataclasses import ChatMessage, Document, GeneratedAnswer


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
    data = {"data": {"key": "value", "more": False}, "type": "test_base_serialization.CustomClass"}

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


def test_serialize_value_with_schema():
    data = {
        "numbers": 1,
        "messages": [ChatMessage.from_user(text="Hello, world!")],
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
    result = serialize_value_with_schema(data)
    assert result == {
        "serialization_schema": {
            "numbers": {"type": "integer"},
            "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
            "user_id": {"type": "string"},
            "dict_of_lists": {"type": "object"},
            "documents": {"type": "array", "items": {"type": "haystack.dataclasses.document.Document"}},
            "list_of_dicts": {"type": "array", "items": {"type": "string"}},
            "answers": {"type": "array", "items": {"type": "haystack.dataclasses.answer.GeneratedAnswer"}},
        },
        "serialized_data": {
            "numbers": 1,
            "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]}],
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
            "numbers": {"type": "integer"},
            "messages": {"type": "array", "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"}},
            "user_id": {"type": "string"},
            "dict_of_lists": {"type": "object"},
            "documents": {"type": "array", "items": {"type": "haystack.dataclasses.document.Document"}},
            "list_of_dicts": {"type": "array", "items": {"type": "string"}},
            "answers": {"type": "array", "items": {"type": "haystack.dataclasses.answer.GeneratedAnswer"}},
        },
        "serialized_data": {
            "numbers": 1,
            "messages": [{"role": "user", "meta": {}, "name": None, "content": [{"text": "Hello, world!"}]}],
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

    result = deserialize_value_with_schema(serialized__data)
    assert result["numbers"] == 1
    assert isinstance(result["messages"][0], ChatMessage)
    assert result["messages"][0].text == "Hello, world!"
    assert result["user_id"] == "123"
    assert result["dict_of_lists"] == {"numbers": [1, 2, 3]}
    assert isinstance(result["documents"][0], Document)
    assert result["documents"][0].content == "Hello, world!"
    assert isinstance(result["answers"][0], GeneratedAnswer)


def test_serialize_value_with_custom_class_type():
    custom_type = CustomClass()
    data = {"numbers": 1, "custom_type": custom_type}
    result = serialize_value_with_schema(data)
    assert result == {
        "serialization_schema": {
            "numbers": {"type": "integer"},
            "custom_type": {"type": "test_base_serialization.CustomClass"},
        },
        "serialized_data": {"numbers": 1, "custom_type": {"key": "value", "more": False}},
    }


def test_deserialize_value_with_custom_class_type():
    serialized_data = {
        "serialization_schema": {
            "numbers": {"type": "integer"},
            "custom_type": {"type": "test_base_serialization.CustomClass"},
        },
        "serialized_data": {"numbers": 1, "custom_type": {"key": "value", "more": False}},
    }
    result = deserialize_value_with_schema(serialized_data)
    assert result["numbers"] == 1
    assert isinstance(result["custom_type"], CustomClass)
