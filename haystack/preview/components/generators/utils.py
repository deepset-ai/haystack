import inspect
import sys
from typing import Optional, Callable

from haystack.preview import DeserializationError
from haystack.preview.dataclasses import StreamingChunk


def default_streaming_callback(chunk: StreamingChunk) -> None:
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
    module = inspect.getmodule(streaming_callback)

    # Get the full package path of the function
    if module is not None:
        full_path = f"{module.__name__}.{streaming_callback.__name__}"
    else:
        full_path = streaming_callback.__name__
    return full_path


def deserialize_callback_handler(callback_name: str) -> Optional[Callable[[StreamingChunk], None]]:
    """
    Deserializes the streaming callback handler.
    :param callback_name: The full path of the streaming callback handler function
    :return: The streaming callback handler function
    :raises DeserializationError: If the streaming callback handler function cannot be found
    """
    parts = callback_name.split(".")
    module_name = ".".join(parts[:-1])
    function_name = parts[-1]
    module = sys.modules.get(module_name, None)
    if not module:
        raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
    streaming_callback = getattr(module, function_name, None)
    if not streaming_callback:
        raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
    return streaming_callback
