import pytest

from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.generators.utils import serialize_callback_handler, deserialize_callback_handler


# streaming callback needs to be on module level
def streaming_callback(chunk):
    pass


def test_callback_handler_serialization():
    result = serialize_callback_handler(streaming_callback)
    assert result == "test_utils.streaming_callback"


def test_callback_handler_serialization_non_local():
    result = serialize_callback_handler(print_streaming_chunk)
    assert result == "haystack.components.generators.utils.print_streaming_chunk"


def test_callback_handler_deserialization():
    result = serialize_callback_handler(streaming_callback)
    fn = deserialize_callback_handler(result)

    assert fn is streaming_callback


def test_callback_handler_deserialization_non_local():
    result = serialize_callback_handler(print_streaming_chunk)
    fn = deserialize_callback_handler(result)

    assert fn is print_streaming_chunk
