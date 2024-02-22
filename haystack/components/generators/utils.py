from typing import Optional, Callable

from haystack.dataclasses import StreamingChunk
from haystack.utils import serialize_callable, deserialize_callable


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.
    Prints the tokens of the first completion to stdout as soon as they are received
    """
    print(chunk.content, flush=True, end="")


def serialize_callback_handler(streaming_callback: Callable[[StreamingChunk], None]) -> str:
    """
    Serializes the streaming callback handler.
    :param streaming_callback: The streaming callback handler function
    :return: The full path of the streaming callback handler function
    """
    return serialize_callable(streaming_callback)


def deserialize_callback_handler(callback_name: str) -> Optional[Callable[[StreamingChunk], None]]:
    """
    Deserializes the streaming callback handler.
    :param callback_name: The full path of the streaming callback handler function
    :return: The streaming callback handler function
    :raises DeserializationError: If the streaming callback handler function cannot be found
    """
    return deserialize_callable(callback_name)
