import inspect
import sys
from typing import Callable, Optional

from haystack import DeserializationError


def serialize_callable(callable_handle: Callable) -> str:
    """
    Serializes a callable to its full path.
    :param callable_handle: The callable to serialize
    :return: The full path of the callable
    """
    module = inspect.getmodule(callable_handle)

    # Get the full package path of the function
    if module is not None:
        full_path = f"{module.__name__}.{callable_handle.__name__}"
    else:
        full_path = callable_handle.__name__
    return full_path


def deserialize_callable(callable_handle: str) -> Optional[Callable]:
    """
    Deserializes a callable given its full import path as a string.
    :param callable_handle: The full path of the callable_handle
    :return: The callable
    :raises DeserializationError: If the callable cannot be found
    """
    parts = callable_handle.split(".")
    module_name = ".".join(parts[:-1])
    function_name = parts[-1]
    module = sys.modules.get(module_name, None)
    if not module:
        raise DeserializationError(f"Could not locate the module of the callable: {module_name}")
    streaming_callback = getattr(module, function_name, None)
    if not streaming_callback:
        raise DeserializationError(f"Could not locate the callable: {function_name}")
    return streaming_callback
