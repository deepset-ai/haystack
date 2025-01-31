# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Callable

from haystack.core.errors import DeserializationError, SerializationError
from haystack.utils.type_serialization import thread_safe_import


def serialize_callable(callable_handle: Callable) -> str:
    """
    Serializes a callable to its full path.

    :param callable_handle: The callable to serialize
    :return: The full path of the callable
    """
    try:
        full_arg_spec = inspect.getfullargspec(callable_handle)
        is_instance_method = bool(full_arg_spec.args and full_arg_spec.args[0] == "self")
    except TypeError:
        is_instance_method = False
    if is_instance_method:
        raise SerializationError("Serialization of instance methods is not supported.")

    # __qualname__ contains the fully qualified path we need for classmethods and staticmethods
    qualname = getattr(callable_handle, "__qualname__", "")
    if "<lambda>" in qualname:
        raise SerializationError("Serialization of lambdas is not supported.")
    if "<locals>" in qualname:
        raise SerializationError("Serialization of nested functions is not supported.")

    name = qualname or callable_handle.__name__

    # Get the full package path of the function
    module = inspect.getmodule(callable_handle)
    if module is not None:
        full_path = f"{module.__name__}.{name}"
    else:
        full_path = name
    return full_path


def deserialize_callable(callable_handle: str) -> Callable:
    """
    Deserializes a callable given its full import path as a string.

    :param callable_handle: The full path of the callable_handle
    :return: The callable
    :raises DeserializationError: If the callable cannot be found
    """
    parts = callable_handle.split(".")

    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            mod: Any = thread_safe_import(module_name)
        except Exception:
            # keep reducing i until we find a valid module import
            continue

        attr_value = mod
        for part in parts[i:]:
            try:
                attr_value = getattr(attr_value, part)
            except AttributeError as e:
                raise DeserializationError(f"Could not find attribute '{part}' in {attr_value.__name__}") from e

        # when the attribute is a classmethod, we need the underlying function
        if isinstance(attr_value, (classmethod, staticmethod)):
            attr_value = attr_value.__func__

        if not callable(attr_value):
            raise DeserializationError(f"The final attribute is not callable: {attr_value}")

        return attr_value

    # Fallback if we never find anything
    raise DeserializationError(f"Could not import '{callable_handle}' as a module or callable.")
