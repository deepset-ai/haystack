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
    _check_not_denied_callable,
    _check_not_deserialization_internal,
    _check_resolved_module_allowed,
    _check_traversable_attribute,
    _is_denied_builtin,
    _is_module_allowed,
    mark_deserialization_internal,
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


@mark_deserialization_internal
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
    from haystack.hooks.from_function import FunctionHook
    from haystack.tools.tool import Tool

    parts = callable_handle.split(".")

    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        # Only import modules that are on the allowlist. Gating the import (rather than a mere
        # string prefix of the handle) means a disallowed module is never imported for its
        # side effects, and the resolver can only ever start from a trusted module. Shorter
        # prefixes are tried in turn, so `json.dumps` still resolves when `json` is allowed.
        if not _is_module_allowed(module_name):
            continue
        try:
            mod: Any = thread_safe_import(module_name)
        except Exception:
            # keep reducing i until we find a valid module import
            continue

        attr_value = mod
        for part in parts[i:]:
            # A handle legitimately walks `module.Class.method`, never into an object's internals.
            # Refuse dunder/frame attributes (`__globals__`, `__dict__`, `__class__`, ...) before the
            # getattr: `<func>.__globals__` yields a live module namespace (a gateway to the allowlist
            # state and to `__builtins__`/`eval`) even though the traversal never leaves an allowlisted
            # module, so neither the module allowlist nor the resolved-object checks below would catch it.
            _check_traversable_attribute(part, callable_handle)
            try:
                attr_value = getattr(attr_value, part)
            except AttributeError as e:
                container = getattr(attr_value, "__name__", type(attr_value).__name__)
                raise DeserializationError(f"Could not find attribute '{part}' in {container}") from e
            # A crafted handle can walk through an object re-exported from an unallowlisted module and then reach a
            # final callable whose own module is allowlisted. For example, an allowlisted Haystack module re-exports
            # `rich.console.Console`; walking through that class to `Console._environ.update` ends at
            # `collections.abc.MutableMapping.update`, hiding the unallowlisted `rich` hop from the final check below.
            # Validate every object reached during traversal so no intermediate hop can escape the allowlist.
            _check_resolved_module_allowed(attr_value, declared_module=module_name)

        # when the attribute is a classmethod, we need the underlying function
        if isinstance(attr_value, (classmethod, staticmethod)):
            attr_value = attr_value.__func__

        # Handle the case where @tool decorator replaced the function with a Tool object
        if isinstance(attr_value, Tool):
            attr_value = attr_value.function or attr_value.async_function

        # Handle the case where @hook decorator replaced the function with a FunctionHook object
        if isinstance(attr_value, FunctionHook):
            attr_value = attr_value.function or attr_value.async_function

        if not callable(attr_value):
            raise DeserializationError(f"The final attribute is not callable: {attr_value}")

        # Final defense: gate on the module the resolved callable actually comes from, not on the
        # declared handle. This catches a dangerous callable bound as a plain (non-module) attribute
        # of an allowlisted object, which the module-walk check above would not see. `module_name`
        # is the allowlisted module we resolved from, so a private C accelerator backing it (e.g.
        # `operator.add` -> `_operator`) is still accepted.
        _check_resolved_module_allowed(attr_value, declared_module=module_name)

        # `builtins` is on the allowlist (for `builtins.print` etc.), so the module check
        # above does not stop dangerous builtins like `eval`/`exec` from resolving here. Block them.
        _check_not_denied_builtin(attr_value, callable_handle)

        # The module check also does not stop import primitives that live inside an allowlisted
        # namespace (e.g. `haystack...thread_safe_import`), which are gateways to code execution
        # equivalent to the denied builtin `__import__`. Block them too.
        _check_not_denied_callable(attr_value, callable_handle)

        # Refuse the deserializer's own machinery — the allowlist-administration function
        # (`allow_deserialization_module`) and the resolution helpers (`deserialize_callable`,
        # `deserialize_type`, `import_class_by_name`). They live in the allowlisted `haystack`
        # namespace, so the module checks above admit them, but resolving them from serialized data
        # lets a hostile pipeline register them as Jinja custom filters, disarm the allowlist with
        # `'*'`, and then resolve and invoke arbitrary callables such as `os.system`.
        _check_not_deserialization_internal(attr_value, callable_handle)

        return attr_value

    # Nothing on the allowlist was importable. Surface the standard allowlist error when the
    # top-level module is untrusted; otherwise report a plain resolution failure.
    _check_module_allowed(callable_handle)
    raise DeserializationError(f"Could not import '{callable_handle}' as a module or callable.")
