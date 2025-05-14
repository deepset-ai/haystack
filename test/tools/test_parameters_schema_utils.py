# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import List

from haystack.dataclasses import ByteStream, ChatMessage, Document, TextContent, ToolCall, ToolCallResult
from pydantic import Field, create_model
from haystack.tools.parameters_schema_utils import _resolve_type
from haystack.tools.from_function import _remove_title_from_schema


BYTE_STREAM_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {"type": "string", "description": "The binary data stored in Bytestream.", "format": "binary"},
        "meta": {
            "type": "object",
            "default": {},
            "description": "Additional metadata to be stored with the ByteStream.",
            "additionalProperties": True,
        },
        "mime_type": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "The mime type of the binary data.",
        },
    },
    "required": ["data"],
}

SPARSE_EMBEDDING_SCHEMA = {
    "type": "object",
    "properties": {
        "indices": {
            "type": "array",
            "description": "List of indices of non-zero elements in the embedding.",
            "items": {"type": "integer"},
        },
        "values": {
            "type": "array",
            "description": "List of values of non-zero elements in the embedding.",
            "items": {"type": "number"},
        },
    },
    "required": ["indices", "values"],
}

DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique identifier for the document. When not set, it's generated based on the Document fields' values.",
            "default": "",
        },
        "content": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Text of the document, if the document contains text.",
        },
        "blob": {
            "anyOf": [{"$ref": "#/$defs/ByteStream"}, {"type": "null"}],
            "default": None,
            "description": "Binary data associated with the document, if the document has any binary data associated with it.",
        },
        "meta": {
            "type": "object",
            "description": "Additional custom metadata for the document. Must be JSON-serializable.",
            "default": {},
            "additionalProperties": True,
        },
        "score": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "default": None,
            "description": "Score of the document. Used for ranking, usually assigned by retrievers.",
        },
        "embedding": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}],
            "default": None,
            "description": "dense vector representation of the document.",
        },
        "sparse_embedding": {
            "anyOf": [{"$ref": "#/$defs/SparseEmbedding"}, {"type": "null"}],
            "default": None,
            "description": "sparse vector representation of the document.",
        },
    },
}

TEXT_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "The text content of the message."}},
    "required": ["text"],
}

TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "description": "The name of the Tool to call."},
        "arguments": {
            "type": "object",
            "description": "The arguments to call the Tool with.",
            "additionalProperties": True,
        },
        "id": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "The ID of the Tool call.",
        },
    },
    "required": ["tool_name", "arguments"],
}

TOOL_CALL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string", "description": "The result of the Tool invocation."},
        "origin": {"$ref": "#/$defs/ToolCall", "description": "The Tool call that produced this result."},
        "error": {"type": "boolean", "description": "Whether the Tool invocation resulted in an error."},
    },
    "required": ["result", "origin", "error"],
}

CHAT_ROLE_SCHEMA = {
    "description": "Enumeration representing the roles within a chat.",
    "enum": ["user", "system", "assistant", "tool"],
    "type": "string",
}

CHAT_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"$ref": "#/$defs/ChatRole", "description": "Field 'role' of 'ChatMessage'."},
        "content": {
            "type": "array",
            "description": "Field 'content' of 'ChatMessage'.",
            "items": {
                "anyOf": [
                    {"$ref": "#/$defs/TextContent"},
                    {"$ref": "#/$defs/ToolCall"},
                    {"$ref": "#/$defs/ToolCallResult"},
                ]
            },
        },
        "name": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Field 'name' of 'ChatMessage'.",
        },
        "meta": {
            "type": "object",
            "description": "Field 'meta' of 'ChatMessage'.",
            "default": {},
            "additionalProperties": True,
        },
    },
    "required": ["role", "content"],
}


@pytest.mark.parametrize(
    "python_type, description, expected_schema, expected_defs_schema",
    [
        (
            ByteStream,
            "A byte stream",
            {"$ref": "#/$defs/ByteStream", "description": "A byte stream"},
            {"ByteStream": BYTE_STREAM_SCHEMA},
        ),
        (
            Document,
            "A document",
            {"$ref": "#/$defs/Document", "description": "A document"},
            {"Document": DOCUMENT_SCHEMA, "SparseEmbedding": SPARSE_EMBEDDING_SCHEMA, "ByteStream": BYTE_STREAM_SCHEMA},
        ),
        (
            TextContent,
            "A text content",
            {"$ref": "#/$defs/TextContent", "description": "A text content"},
            {"TextContent": TEXT_CONTENT_SCHEMA},
        ),
        (
            ToolCall,
            "A tool call",
            {"$ref": "#/$defs/ToolCall", "description": "A tool call"},
            {"ToolCall": TOOL_CALL_SCHEMA},
        ),
        (
            ToolCallResult,
            "A tool call result",
            {"$ref": "#/$defs/ToolCallResult", "description": "A tool call result"},
            {"ToolCallResult": TOOL_CALL_RESULT_SCHEMA, "ToolCall": TOOL_CALL_SCHEMA},
        ),
        (
            ChatMessage,
            "A chat message",
            {"$ref": "#/$defs/ChatMessage", "description": "A chat message"},
            {
                "ChatMessage": CHAT_MESSAGE_SCHEMA,
                "TextContent": TEXT_CONTENT_SCHEMA,
                "ToolCall": TOOL_CALL_SCHEMA,
                "ToolCallResult": TOOL_CALL_RESULT_SCHEMA,
                "ChatRole": CHAT_ROLE_SCHEMA,
            },
        ),
        (
            List[Document],
            "A list of documents",
            {"type": "array", "description": "A list of documents", "items": {"$ref": "#/$defs/Document"}},
            {"Document": DOCUMENT_SCHEMA, "SparseEmbedding": SPARSE_EMBEDDING_SCHEMA, "ByteStream": BYTE_STREAM_SCHEMA},
        ),
        (
            List[ChatMessage],
            "A list of chat messages",
            {"type": "array", "description": "A list of chat messages", "items": {"$ref": "#/$defs/ChatMessage"}},
            {
                "ChatMessage": CHAT_MESSAGE_SCHEMA,
                "TextContent": TEXT_CONTENT_SCHEMA,
                "ToolCall": TOOL_CALL_SCHEMA,
                "ToolCallResult": TOOL_CALL_RESULT_SCHEMA,
                "ChatRole": CHAT_ROLE_SCHEMA,
            },
        ),
    ],
)
def test_create_parameters_schema_haystack_dataclasses(python_type, description, expected_schema, expected_defs_schema):
    resolved_type = _resolve_type(python_type)
    fields = {"input_name": (resolved_type, Field(default=..., description=description))}
    model = create_model("run", __doc__="A test function", **fields)
    parameters_schema = model.model_json_schema()
    _remove_title_from_schema(parameters_schema)

    defs_schema = parameters_schema["$defs"]
    assert defs_schema == expected_defs_schema

    property_schema = parameters_schema["properties"]["input_name"]
    assert property_schema == expected_schema
