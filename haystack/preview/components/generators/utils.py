import sys
from typing import Optional, Callable

from haystack.preview import DeserializationError
from haystack.preview.dataclasses import StreamingChunk


def serialize_callback_handler(streaming_callback: Callable[[StreamingChunk], None]) -> str:
    """
    Serialize the streaming callback handler.
    """
    module = streaming_callback.__module__
    if module == "builtins":
        callback_name = streaming_callback.__name__
    else:
        callback_name = f"{module}.{streaming_callback.__name__}"
    return callback_name


def deserialize_callback_handler(callback_name: str) -> Optional[Callable[[StreamingChunk], None]]:
    """
    Deserialize the streaming callback handler.
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
