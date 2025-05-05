import pytest
from typing import Any, Dict, List, Optional, Union

from haystack.dataclasses import ByteStream, ChatMessage, Document
from haystack.tools.property_schema_utils import _create_property_schema


def remove_item(d, key):
    new_d = d.copy()
    new_d.pop(key, None)
    return new_d


BYTE_STREAM_SCHEMA = {
    "type": "object",
    "description": "A byte stream",
    "properties": {
        "data": {"type": "string", "description": "Field 'data' of 'ByteStream'."},
        "meta": {"type": "object", "description": "Field 'meta' of 'ByteStream'.", "additionalProperties": True},
        "mime_type": {
            "oneOf": [{"type": "string"}, {"type": "null"}],
            "description": "Field 'mime_type' of 'ByteStream'.",
        },
    },
}

DOCUMENT_SCHEMA = {
    "type": "object",
    "description": "A document",
    "properties": {
        "id": {"type": "string", "description": "Field 'id' of 'Document'."},
        "content": {"oneOf": [{"type": "string"}, {"type": "null"}], "description": "Field 'content' of 'Document'."},
        "blob": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Field 'data' of 'ByteStream'."},
                        "meta": {
                            "type": "object",
                            "description": "Field 'meta' of 'ByteStream'.",
                            "additionalProperties": True,
                        },
                        "mime_type": {
                            "oneOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Field 'mime_type' of 'ByteStream'.",
                        },
                    },
                },
                {"type": "null"},
            ],
            "description": "Field 'blob' of 'Document'.",
        },
        "meta": {"type": "object", "description": "Field 'meta' of 'Document'.", "additionalProperties": True},
        "score": {"oneOf": [{"type": "number"}, {"type": "null"}], "description": "Field 'score' of 'Document'."},
        "embedding": {
            "oneOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}],
            "description": "Field 'embedding' of 'Document'.",
        },
        "sparse_embedding": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "indices": {
                            "type": "array",
                            "description": "Field 'indices' of 'SparseEmbedding'.",
                            "items": {"type": "integer"},
                        },
                        "values": {
                            "type": "array",
                            "description": "Field 'values' of 'SparseEmbedding'.",
                            "items": {"type": "number"},
                        },
                    },
                },
                {"type": "null"},
            ],
            "description": "Field 'sparse_embedding' of 'Document'.",
        },
    },
}

CHAT_MESSAGE_SCHEMA = {
    "type": "object",
    "description": "A chat message",
    "properties": {
        "_role": {"type": "string", "description": "Field '_role' of 'ChatMessage'."},
        "_content": {"type": "string", "description": "Field '_content' of 'ChatMessage'."},
        "_name": {"oneOf": [{"type": "string"}, {"type": "null"}], "description": "Field '_name' of 'ChatMessage'."},
        "_meta": {"type": "object", "description": "Field '_meta' of 'ChatMessage'.", "additionalProperties": True},
    },
}


@pytest.mark.parametrize(
    "python_type, expected_schema",
    [
        (str, {"type": "string", "description": ""}),
        (int, {"type": "integer", "description": ""}),
        (float, {"type": "number", "description": ""}),
        (bool, {"type": "boolean", "description": ""}),
        (list, {"type": "array", "description": ""}),
        (dict, {"type": "object", "description": "", "additionalProperties": True}),
    ],
)
def test_create_property_schema_bare_types(python_type, expected_schema):
    """
    Test the _create_property_schema function with various Python types.
    """
    schema = _create_property_schema(python_type, "")
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            Optional[str],
            "An optional string",
            {"oneOf": [{"type": "string"}, {"type": "null"}], "description": "An optional string"},
        ),
        (
            Optional[int],
            "An optional integer",
            {"oneOf": [{"type": "integer"}, {"type": "null"}], "description": "An optional integer"},
        ),
        (
            Optional[float],
            "An optional float",
            {"oneOf": [{"type": "number"}, {"type": "null"}], "description": "An optional float"},
        ),
        (
            Optional[bool],
            "An optional boolean",
            {"oneOf": [{"type": "boolean"}, {"type": "null"}], "description": "An optional boolean"},
        ),
        (
            Optional[list],
            "An optional list",
            {"oneOf": [{"type": "array"}, {"type": "null"}], "description": "An optional list"},
        ),
        (
            Optional[dict],
            "An optional dict",
            {
                "oneOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}],
                "description": "An optional dict",
            },
        ),
    ],
)
def test_create_property_schema_optional_types(python_type, description, expected_schema):
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            List[str],
            "A list of strings",
            {"type": "array", "description": "A list of strings", "items": {"type": "string"}},
        ),
        (
            List[int],
            "A list of integers",
            {"type": "array", "description": "A list of integers", "items": {"type": "integer"}},
        ),
        (
            List[float],
            "A list of floats",
            {"type": "array", "description": "A list of floats", "items": {"type": "number"}},
        ),
        (
            List[bool],
            "A list of booleans",
            {"type": "array", "description": "A list of booleans", "items": {"type": "boolean"}},
        ),
    ],
)
def test_create_property_schema_list_of_types(python_type, description, expected_schema):
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            Union[str, int],
            "A union of string and integer",
            {"description": "A union of string and integer", "oneOf": [{"type": "string"}, {"type": "integer"}]},
        ),
        (
            Union[str, float],
            "A union of string and float",
            {"description": "A union of string and float", "oneOf": [{"type": "string"}, {"type": "number"}]},
        ),
    ],
)
def test_create_property_schema_union_type(python_type, description, expected_schema):
    """
    Test the _create_property_schema function with union types.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (ByteStream, "A byte stream", BYTE_STREAM_SCHEMA),
        (Document, "A document", DOCUMENT_SCHEMA),
        (ChatMessage, "A chat message", CHAT_MESSAGE_SCHEMA),
        (
            List[Document],
            "A list of documents",
            {
                "type": "array",
                "description": "A list of documents",
                "items": remove_item(DOCUMENT_SCHEMA, "description"),
            },
        ),
        (
            List[ChatMessage],
            "A list of chat messages",
            {
                "type": "array",
                "description": "A list of chat messages",
                "items": remove_item(CHAT_MESSAGE_SCHEMA, "description"),
            },
        ),
        (
            Optional[List[ChatMessage]],
            "An optional list of chat messages",
            {
                "oneOf": [
                    {"type": "array", "items": remove_item(CHAT_MESSAGE_SCHEMA, "description")},
                    {"type": "null"},
                ],
                "description": "An optional list of chat messages",
            },
        ),
        (
            List[List[Document]],
            "A list of lists of documents",
            {
                "type": "array",
                "items": {"type": "array", "items": remove_item(DOCUMENT_SCHEMA, "description")},
                "description": "A list of lists of documents",
            },
        ),
    ],
)
def test_create_property_schema_haystack_dataclasses(python_type, description, expected_schema):
    """
    Test the _create_property_schema function with haystack dataclasses.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            Union[Dict[str, Any], List[Dict[str, Any]]],
            "Often found as the runtime param `meta` in our components",
            {
                "description": "Often found as the runtime param `meta` in our components",
                "oneOf": [
                    {"type": "object", "additionalProperties": True},
                    {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                ],
            },
        )
    ],
)
def test_create_property_schema_complex_types(python_type, description, expected_schema):
    """
    Tests for complex types, especially those found in our pre-built components.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema
