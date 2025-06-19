# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import ComponentInfo, FinishReason, StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult


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
            tool_call=ToolCallDelta(id="123", tool_name="test_tool", arguments='{"arg1": "value1"}'),
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


def test_component_info_from_component():
    component = ExampleComponent()
    component_info = ComponentInfo.from_component(component)
    assert component_info.type == "test_streaming_chunk.ExampleComponent"


def test_component_info_from_component_with_name_from_pipeline():
    pipeline = Pipeline()
    component = ExampleComponent()
    pipeline.add_component("pipeline_component", component)
    component_info = ComponentInfo.from_component(component)
    assert component_info.type == "test_streaming_chunk.ExampleComponent"
    assert component_info.name == "pipeline_component"


def test_tool_call_delta():
    tool_call = ToolCallDelta(id="123", tool_name="test_tool", arguments='{"arg1": "value1"}')
    assert tool_call.id == "123"
    assert tool_call.tool_name == "test_tool"
    assert tool_call.arguments == '{"arg1": "value1"}'


def test_tool_call_delta_with_missing_fields():
    with pytest.raises(ValueError):
        _ = ToolCallDelta(id="123")


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


def test_finish_reason_deprecation_warning():
    """Test that accessing finish_reason via meta shows deprecation warning."""
    import warnings

    chunk = StreamingChunk(content="Test content", meta={"finish_reason": "length"})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = chunk.meta.get("finish_reason")

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "finish_reason" in str(w[0].message)
        assert "deprecated" in str(w[0].message)
        assert result == "length"


def test_finish_reason_openai_standard_values():
    """Test that the finish_reason field accepts OpenAI standard values."""
    openai_values = ["stop", "length", "tool_calls", "content_filter"]

    for value in openai_values:
        chunk = StreamingChunk(content="Test content", finish_reason=value)
        assert chunk.finish_reason == value


def test_finish_reason_custom_values():
    """Test that custom finish_reason values still work (for migration compatibility)."""
    custom_values = ["custom_stop", "provider_specific", "unknown"]

    for value in custom_values:
        chunk = StreamingChunk(content="Test content", finish_reason=value)
        assert chunk.finish_reason == value
