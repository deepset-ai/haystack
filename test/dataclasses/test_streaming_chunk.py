# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.dataclasses import StreamingChunk, FinishReason, StreamingToolCall, ToolResult, Usage


def test_create_chunk_with_content_and_metadata():
    chunk = StreamingChunk(content="Test content", meta={"key": "value"})

    assert chunk.content == "Test content"
    assert chunk.meta == {"key": "value"}


def test_create_chunk_with_only_content():
    chunk = StreamingChunk(content="Test content")

    assert chunk.content == "Test content"
    assert chunk.meta == {}


def test_access_content():
    chunk = StreamingChunk(content="Test content", meta={"key": "value"})
    assert chunk.content == "Test content"


def test_create_chunk_with_empty_content():
    chunk = StreamingChunk(content="")
    assert chunk.content == ""
    assert chunk.meta == {}


def test_create_chunk_with_tool_calls():
    tool_call = StreamingToolCall(
        id="call_123", tool_name="test_function", arguments={"arg1": "value1"}, type="function"
    )
    chunk = StreamingChunk(content="Test content", tool_calls=[tool_call])

    assert chunk.content == "Test content"
    assert len(chunk.tool_calls) == 1
    assert chunk.tool_calls[0].id == "call_123"
    assert chunk.tool_calls[0].tool_name == "test_function"
    assert chunk.tool_calls[0].arguments == {"arg1": "value1"}
    assert chunk.tool_calls[0].type == "function"


def test_create_chunk_with_tool_results():
    tool_result = ToolResult(tool_call_id="call_123", name="test_function", content="result", error=False)
    chunk = StreamingChunk(content="Test content", tool_results=[tool_result])

    assert chunk.content == "Test content"
    assert len(chunk.tool_results) == 1
    assert chunk.tool_results[0].tool_call_id == "call_123"
    assert chunk.tool_results[0].name == "test_function"
    assert chunk.tool_results[0].content == "result"
    assert chunk.tool_results[0].error is False


def test_create_chunk_with_finish_reason():
    chunk = StreamingChunk(content="Test content", finish_reason=FinishReason.STOP)

    assert chunk.content == "Test content"
    assert chunk.finish_reason == FinishReason.STOP


def test_create_chunk_with_usage():
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    chunk = StreamingChunk(content="Test content", usage=usage)

    assert chunk.content == "Test content"
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 20
    assert chunk.usage.total_tokens == 30


def test_create_chunk_with_all_fields():
    tool_call = StreamingToolCall(
        id="call_123", tool_name="test_function", arguments={"arg1": "value1"}, type="function"
    )
    tool_result = ToolResult(tool_call_id="call_123", name="test_function", content="result", error=False)
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    chunk = StreamingChunk(
        content="Test content",
        meta={"key": "value"},
        tool_calls=[tool_call],
        tool_results=[tool_result],
        finish_reason=FinishReason.TOOL_CALLS,
        usage=usage,
    )

    assert chunk.content == "Test content"
    assert chunk.meta == {"key": "value"}
    assert len(chunk.tool_calls) == 1
    assert chunk.tool_calls[0].id == "call_123"
    assert chunk.tool_calls[0].tool_name == "test_function"
    assert chunk.tool_calls[0].arguments == {"arg1": "value1"}
    assert len(chunk.tool_results) == 1
    assert chunk.tool_results[0].tool_call_id == "call_123"
    assert chunk.tool_results[0].name == "test_function"
    assert chunk.tool_results[0].content == "result"
    assert chunk.tool_results[0].error is False
    assert chunk.finish_reason == FinishReason.TOOL_CALLS
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 20
    assert chunk.usage.total_tokens == 30


def test_finish_reason_values():
    assert FinishReason.STOP.value == "stop"
    assert FinishReason.LENGTH.value == "length"
    assert FinishReason.CONTENT_FILTER.value == "content_filter"
    assert FinishReason.TOOL_CALLS.value == "tool_calls"
    assert FinishReason.TOOL_USE.value == "tool_use"
    assert FinishReason.ERROR.value == "error"
