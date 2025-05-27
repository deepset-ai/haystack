# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack.dataclasses import StreamingChunk, ComponentInfo
from unittest.mock import Mock
from haystack.core.component import Component
from haystack import component
from haystack import Pipeline


@component
class TestComponent:
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


def test_component_info_from_component():
    component = TestComponent()
    component_info = ComponentInfo.from_component(component)
    assert component_info.type == "test_streaming_chunk.TestComponent"


def test_component_info_from_component_with_name_from_pipeline():
    pipeline = Pipeline()
    component = TestComponent()
    pipeline.add_component("pipeline_component", component)
    component_info = ComponentInfo.from_component(component)
    assert component_info.type == "test_streaming_chunk.TestComponent"
    assert component_info.name == "pipeline_component"
