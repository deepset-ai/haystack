# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from typing import Any

from haystack import logging
from haystack.core.errors import DeserializationError, SerializationError
from haystack.core.serialization_security import (
    _check_module_allowed,
    _check_not_denied_builtin,
    _is_denied_builtin,
    _is_module_allowed,
)
from haystack.utils.type_serialization import thread_safe_import

logger = logging.getLogger(__name__)


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

    # Serialization succeeds, but a denied builtin (e.g. `eval`) won't reload without `unsafe=True`.
    if _is_denied_builtin(callable_handle):
        logger.warning(
            "Serialized callable '{full_path}' is a builtin that is blocked during deserialization; "
            "the resulting pipeline will only be loadable with unsafe=True.",
            full_path=full_path,
        )

    return full_path


def deserialize_callable(callable_handle: str) -> Callable:
    """
    Deserializes a callable given its full import path as a string.

    Every module path tried during resolution is checked against the
    deserialization allowlist (see `haystack.core.serialization_security`). Callables in modules
    outside the allowlist are rejected with a `DeserializationError` before any import is
    attempted. To allow a third-party module, extend the allowlist via
    `Pipeline.load(..., allowed_modules=[...])`, `allow_deserialization_module(...)`, or the
    `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

    :param callable_handle: The full path of the callable_handle
    :return: The callable
    :raises DeserializationError:
        If the module path is not on the deserialization allowlist, or if the callable cannot
        be found.
    """
    # Import here to avoid circular imports
    from haystack.tools.tool import Tool

    parts = callable_handle.split(".")

    # Allow if any prefix is on the allowlist; checking each one individually would wrongly
    # reject patterns like `j*on` against `json.dumps` (matches `json`, not the full handle).
    if not any(_is_module_allowed(".".join(parts[:i])) for i in range(1, len(parts) + 1)):
        _check_module_allowed(callable_handle)  # raises with the standard help message

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
                container = getattr(attr_value, "__name__", type(attr_value).__name__)
                raise DeserializationError(f"Could not find attribute '{part}' in {container}") from e

        # when the attribute is a classmethod, we need the underlying function
        if isinstance(attr_value, (classmethod, staticmethod)):
            attr_value = attr_value.__func__

        # Handle the case where @tool decorator replaced the function with a Tool object
        if isinstance(attr_value, Tool):
            attr_value = attr_value.function

        if not callable(attr_value):
            raise DeserializationError(f"The final attribute is not callable: {attr_value}")

        # `builtins` is on the allowlist (for `builtins.print` etc.), so the module check
        # above does not stop dangerous builtins like `eval`/`exec` from resolving here. Block them.
        _check_not_denied_builtin(attr_value, callable_handle)

        return attr_value

    # Fallback if we never find anything
    raise DeserializationError(f"Could not import '{callable_handle}' as a module or callable.")
