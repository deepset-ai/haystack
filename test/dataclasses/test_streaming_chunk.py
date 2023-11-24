import pytest

from haystack.preview.dataclasses import StreamingChunk


@pytest.mark.unit
def test_create_chunk_with_content_and_metadata():
    chunk = StreamingChunk(content="Test content", metadata={"key": "value"})

    assert chunk.content == "Test content"
    assert chunk.metadata == {"key": "value"}


@pytest.mark.unit
def test_create_chunk_with_only_content():
    chunk = StreamingChunk(content="Test content")

    assert chunk.content == "Test content"
    assert chunk.metadata == {}


@pytest.mark.unit
def test_access_content():
    chunk = StreamingChunk(content="Test content", metadata={"key": "value"})
    assert chunk.content == "Test content"


@pytest.mark.unit
def test_create_chunk_with_empty_content():
    chunk = StreamingChunk(content="")
    assert chunk.content == ""
    assert chunk.metadata == {}
