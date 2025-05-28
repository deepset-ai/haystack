# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.dataclasses import StreamingChunk, ComponentInfo, ToolCallDelta, ToolCallResult, ToolCall
from haystack import component
from haystack import Pipeline


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
