# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import (
    ComponentInfo,
    ReasoningContent,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
    ToolCallResult,
)


@component
class ExampleComponent:
    def __init__(self):
        self.name = "test_component"

    def run(self) -> str:
        return "Test content"


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


def test_create_chunk_with_all_fields():
    component_info = ComponentInfo(type="test.component", name="test_component")
    chunk = StreamingChunk(content="Test content", meta={"key": "value"}, component_info=component_info)

    assert chunk.content == "Test content"
    assert chunk.meta == {"key": "value"}
    assert chunk.component_info == component_info


def test_create_chunk_with_content_and_tool_call():
    with pytest.raises(ValueError):
        # Can't have content + tool_call at the same time
        StreamingChunk(
            content="Test content",
            meta={"key": "value"},
            tool_calls=[ToolCallDelta(id="123", tool_name="test_tool", arguments='{"arg1": "value1"}', index=0)],
        )


def test_create_chunk_with_content_and_tool_call_result():
    with pytest.raises(ValueError):
        # Can't have content + tool_call_result at the same time
        StreamingChunk(
            content="Test content",
            meta={"key": "value"},
            tool_call_result=ToolCallResult(
                result="output",
                origin=ToolCall(id="123", tool_name="test_tool", arguments={"arg1": "value1"}),
                error=False,
            ),
        )


def test_create_chunk_with_content_and_reasoning():
    with pytest.raises(ValueError, match="Only one of `content`, `tool_call`, `tool_call_result`"):
        StreamingChunk(
            content="Test content", meta={"key": "value"}, reasoning=ReasoningContent(reasoning_text="thinking")
        )


def test_reasoning_and_no_index():
    with pytest.raises(
        ValueError, match="If `tool_call`, `tool_call_result` or `reasoning` is set, `index` must also be set."
    ):
        StreamingChunk(content="", meta={"key": "value"}, reasoning=ReasoningContent(reasoning_text="thinking"))


def test_component_info_from_component():
    component_info = ComponentInfo.from_component(ExampleComponent())
    assert component_info.type == "test_streaming_chunk.ExampleComponent"


def test_component_info_from_component_with_name_from_pipeline():
    pipeline = Pipeline()
    comp = ExampleComponent()
    pipeline.add_component("pipeline_component", comp)
    component_info = ComponentInfo.from_component(comp)
    assert component_info.type == "test_streaming_chunk.ExampleComponent"
    assert component_info.name == "pipeline_component"


def test_tool_call_delta():
    tool_call = ToolCallDelta(id="123", tool_name="test_tool", arguments='{"arg1": "value1"}', index=0)
    assert tool_call.id == "123"
    assert tool_call.tool_name == "test_tool"
    assert tool_call.arguments == '{"arg1": "value1"}'
    assert tool_call.index == 0


def test_create_chunk_with_finish_reason():
    """Test creating a chunk with the new finish_reason field."""
    chunk = StreamingChunk(content="Test content", finish_reason="stop")

    assert chunk.content == "Test content"
    assert chunk.finish_reason == "stop"
    assert chunk.meta == {}


def test_create_chunk_with_finish_reason_and_meta():
    """Test creating a chunk with both finish_reason field and meta."""
    chunk = StreamingChunk(
        content="Test content", finish_reason="stop", meta={"model": "gpt-4", "usage": {"tokens": 10}}
    )

    assert chunk.content == "Test content"
    assert chunk.finish_reason == "stop"
    assert chunk.meta["model"] == "gpt-4"
    assert chunk.meta["usage"]["tokens"] == 10


def test_finish_reason_standard_values():
    """Test all standard finish_reason values including the new Haystack-specific ones."""
    standard_values = ["stop", "length", "tool_calls", "content_filter", "tool_call_results"]

    for value in standard_values:
        chunk = StreamingChunk(content="Test content", finish_reason=value)
        assert chunk.finish_reason == value


def test_finish_reason_tool_call_results():
    """Test specifically the new tool_call_results finish reason."""
    chunk = StreamingChunk(content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"})

    assert chunk.finish_reason == "tool_call_results"
    assert chunk.meta["finish_reason"] == "tool_call_results"
    assert chunk.content == ""


def test_to_dict_tool_call_result():
    """Test the to_dict method for StreamingChunk with tool_call_result."""
    component_info = ComponentInfo.from_component(ExampleComponent())
    tool_call_result = ToolCallResult(
        result="output", origin=ToolCall(id="123", tool_name="test_tool", arguments={"arg1": "value1"}), error=False
    )

    chunk = StreamingChunk(
        content="",
        meta={"key": "value"},
        index=0,
        component_info=component_info,
        tool_call_result=tool_call_result,
        finish_reason="tool_call_results",
    )

    d = chunk.to_dict()

    assert d["content"] == ""
    assert d["meta"] == {"key": "value"}
    assert d["index"] == 0
    assert d["component_info"]["type"] == "test_streaming_chunk.ExampleComponent"
    assert d["tool_call_result"]["result"] == "output"
    assert d["tool_call_result"]["error"] is False
    assert d["tool_call_result"]["origin"]["id"] == "123"
    assert d["tool_call_result"]["origin"]["arguments"]["arg1"] == "value1"
    assert d["finish_reason"] == "tool_call_results"
    assert d["reasoning"] is None


def test_to_dict_tool_calls():
    """Test the to_dict method for StreamingChunk with tool_calls."""
    component_info = ComponentInfo.from_component(ExampleComponent())
    tool_calls = [
        ToolCallDelta(id="123", tool_name="test_tool", arguments='{"arg1": "value1"}', index=0),
        ToolCallDelta(id="456", tool_name="another_tool", arguments='{"arg2": "value2"}', index=1),
    ]

    chunk = StreamingChunk(
        content="",
        meta={"key": "value"},
        index=0,
        component_info=component_info,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )

    d = chunk.to_dict()

    assert d["content"] == ""
    assert d["meta"] == {"key": "value"}
    assert d["index"] == 0
    assert d["component_info"]["type"] == "test_streaming_chunk.ExampleComponent"
    assert len(d["tool_calls"]) == 2
    assert d["tool_calls"][0]["id"] == "123"
    assert d["tool_calls"][0]["index"] == 0
    assert d["tool_calls"][1]["id"] == "456"
    assert d["tool_calls"][1]["index"] == 1
    assert d["finish_reason"] == "tool_calls"
    assert d["reasoning"] is None


def test_to_dict_reasoning():
    """Test the to_dict method for StreamingChunk with reasoning."""
    component_info = ComponentInfo.from_component(ExampleComponent())
    reasoning = ReasoningContent(reasoning_text="thinking", extra={"step": 1})

    chunk = StreamingChunk(
        content="",
        meta={"key": "value"},
        index=0,
        component_info=component_info,
        reasoning=reasoning,
        finish_reason="stop",
    )

    d = chunk.to_dict()

    assert d["content"] == ""
    assert d["meta"] == {"key": "value"}
    assert d["index"] == 0
    assert d["component_info"]["type"] == "test_streaming_chunk.ExampleComponent"
    assert d["reasoning"]["reasoning_text"] == "thinking"
    assert d["reasoning"]["extra"]["step"] == 1
    assert d["finish_reason"] == "stop"
    assert d["tool_calls"] is None
    assert d["tool_call_result"] is None


def test_from_dict_tool_call_result():
    """Test the from_dict method for StreamingChunk with tool_call_result."""
    component_info = {"type": "test_streaming_chunk.ExampleComponent", "name": "test_component"}
    tool_call_result = {
        "result": "output",
        "origin": {"id": "123", "tool_name": "test_tool", "arguments": {"arg1": "value1"}},
        "error": False,
    }

    data = {
        "content": "",
        "meta": {"key": "value"},
        "index": 0,
        "component_info": component_info,
        "tool_call_result": tool_call_result,
        "finish_reason": "tool_call_results",
    }

    chunk = StreamingChunk.from_dict(data)

    assert chunk.content == ""
    assert chunk.meta == {"key": "value"}
    assert chunk.index == 0
    assert chunk.component_info.type == "test_streaming_chunk.ExampleComponent"
    assert chunk.component_info.name == "test_component"
    assert chunk.tool_call_result.result == "output"
    assert chunk.tool_call_result.error is False
    assert chunk.tool_call_result.origin.id == "123"
    assert chunk.reasoning is None


def test_from_dict_tool_calls():
    """Test the from_dict method for StreamingChunk with tool_calls."""
    component_info = {"type": "test_streaming_chunk.ExampleComponent", "name": "test_component"}
    tool_calls = [{"id": "123", "tool_name": "test_tool", "arguments": '{"arg1": "value1"}', "index": 0}]

    data = {
        "content": "",
        "meta": {"key": "value"},
        "index": 0,
        "component_info": component_info,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls",
    }

    chunk = StreamingChunk.from_dict(data)

    assert chunk.content == ""
    assert chunk.meta == {"key": "value"}
    assert chunk.index == 0
    assert chunk.component_info.type == "test_streaming_chunk.ExampleComponent"
    assert chunk.component_info.name == "test_component"
    assert chunk.tool_calls[0].tool_name == "test_tool"
    assert chunk.tool_calls[0].index == 0
    assert chunk.finish_reason == "tool_calls"
    assert chunk.reasoning is None


def test_from_dict_reasoning():
    """Test the from_dict method for StreamingChunk with reasoning."""
    component_info = {"type": "test_streaming_chunk.ExampleComponent", "name": "test_component"}
    reasoning = {"reasoning_text": "thinking", "extra": {"step": 1}}

    data = {
        "content": "",
        "meta": {"key": "value"},
        "index": 0,
        "component_info": component_info,
        "reasoning": reasoning,
        "finish_reason": "stop",
    }

    chunk = StreamingChunk.from_dict(data)

    assert chunk.content == ""
    assert chunk.meta == {"key": "value"}
    assert chunk.index == 0
    assert chunk.component_info.type == "test_streaming_chunk.ExampleComponent"
    assert chunk.component_info.name == "test_component"
    assert chunk.reasoning.reasoning_text == "thinking"
    assert chunk.reasoning.extra["step"] == 1
    assert chunk.finish_reason == "stop"
    assert chunk.tool_calls is None
    assert chunk.tool_call_result is None
