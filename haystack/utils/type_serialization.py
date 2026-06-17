# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import inspect
import sys
import typing
from threading import Lock
from types import GenericAlias, ModuleType, NoneType, UnionType
from typing import Any, Union, get_args

from haystack.core.errors import DeserializationError
from haystack.core.serialization_security import _check_builtin_is_type, _check_module_allowed

_import_lock = Lock()


def _is_union_type(target: Any) -> bool:
    """
    Check if target is a Union type.

    This handles both `typing.Union[X, Y]` and `X | Y` syntax from PEP 604,
    including parameterized types like `Optional[str]`.
    """
    if target is Union or target is UnionType:
        return True
    origin = typing.get_origin(target)
    return origin is Union or origin is UnionType


def _build_pep604_union_type(types: list[type | UnionType]) -> type | UnionType:
    """Build a union type from a list of types using PEP 604 syntax (X | Y)."""
    result = types[0]
    for t in types[1:]:
        result = result | t
    return result


def serialize_type(target: Any) -> str:
    """
    Serializes a type or an instance to its string representation, including the module name.

    This function handles types, instances of types, and special typing objects.
    It assumes that non-typing objects will have a '__name__' attribute.

    :param target:
        The object to serialize, can be an instance or a type.
    :return:
        The string representation of the type.
    """
    if target is NoneType:
        return "None"

    args = get_args(target)

    if isinstance(target, UnionType):
        return " | ".join([serialize_type(a) for a in args])

    name = getattr(target, "__name__", str(target))
    if name.startswith("typing."):
        name = name[7:]
    if "[" in name:
        name = name.split("[")[0]

    # Get module name
    module = inspect.getmodule(target)
    module_name = ""
    # We omit the module name for builtins to not clutter the output
    if module and hasattr(module, "__name__") and module.__name__ != "builtins":
        module_name = f"{module.__name__}"

    if args:
        # For typing generics, convert PEP 604 union types (X | Y) to typing.Union when serializing.
        # This avoids issues with Python's internal cache, where List[Union[str, int]] and List[str | int] are treated
        # as the same key. GenericAlias (builtins like list[...]) can keep the PEP 604 syntax.
        is_typing_generic = not isinstance(target, GenericAlias)
        # Optional[X] is normalized by Python to Union[X, None]; the trailing None is already implied by the
        # "Optional" name, so we drop it. For any other generic (e.g. Dict[str, None], Tuple[int, None] or a
        # Union with more than two members) NoneType is a regular argument and must be kept.
        skip_nonetype = name == "Optional"
        args_str = ", ".join(
            serialize_type(Union[tuple(get_args(a))] if is_typing_generic and isinstance(a, UnionType) else a)  # noqa: UP007
            for a in args
            if not (skip_nonetype and a is NoneType)
        )
        return f"{module_name}.{name}[{args_str}]" if module_name else f"{name}[{args_str}]"

    return f"{module_name}.{name}" if module_name else f"{name}"


def _parse_generic_args(args_str: str) -> list[str]:
    args = []
    bracket_count = 0
    current_arg = ""

    for char in args_str:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1

        if char == "," and bracket_count == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char

    if current_arg:
        args.append(current_arg.strip())

    return args


def _parse_pep604_union_args(union_str: str) -> list[str]:
    """
    Parse a PEP 604 union string (e.g., "str | int | None") into individual type strings.

    Handles nested generics properly, e.g., "list[str] | dict[str, int] | None".

    :param union_str: The union string to parse
    :returns: A list of individual type strings
    """
    args = []
    bracket_count = 0
    current_arg = ""

    for char in union_str:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1

        if char == "|" and bracket_count == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args


def deserialize_type(type_str: str) -> Any:
    """
    Deserializes a type given its full import path as a string, including nested generic types.

    This function will dynamically import the module if it's not already imported
    and then retrieve the type object from it. It also handles nested generic types like
    `list[dict[int, str]]`.

    Every module path with a `.` prefix is checked against the deserialization
    allowlist (see `haystack.core.serialization_security`) before being imported. Modules outside
    the allowlist are rejected with a `DeserializationError`. Builtin and `typing`/`collections`
    names without a module prefix bypass this check.

    :param type_str:
        The string representation of the type's full import path.
    :returns:
        The deserialized type object.
    :raises DeserializationError:
        If the module is not on the deserialization allowlist, or if the type cannot be
        deserialized due to a missing module or type.
    """
    # Handle PEP 604 union syntax at the top level (e.g., "str | int", "str | None")
    pep604_union_args = _parse_pep604_union_args(type_str)
    if len(pep604_union_args) > 1:
        deserialized_args = [deserialize_type(arg) for arg in pep604_union_args]
        return _build_pep604_union_type(deserialized_args)

    # Handle generics (including Union[X, Y])
    if "[" in type_str and type_str.endswith("]"):
        main_type_str, generics_str = type_str.split("[", 1)
        generics_str = generics_str[:-1]

        main_type = deserialize_type(main_type_str)
        generic_args = [deserialize_type(arg) for arg in _parse_generic_args(generics_str)]

        # Reconstruct
        try:
            return main_type[tuple(generic_args) if len(generic_args) > 1 else generic_args[0]]
        except (TypeError, AttributeError) as e:
            raise DeserializationError(f"Could not apply arguments {generic_args} to type {main_type}") from e

    # Handle non-generic types
    # First, check if there's a module prefix
    if "." in type_str:
        parts = type_str.split(".")
        module_name = ".".join(parts[:-1])
        type_name = parts[-1]

        _check_module_allowed(module_name)

        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = thread_safe_import(module_name)
            except ImportError as e:
                raise DeserializationError(f"Could not import the module: {module_name}") from e

        # Get the class from the module
        if hasattr(module, type_name):
            resolved = getattr(module, type_name)
            # `builtins` is on the allowlist; a type annotation must resolve to an actual type. This
            # lets builtin types through (e.g. `builtins.memoryview`) while rejecting builtin
            # functions like `builtins.eval`.
            if module_name == "builtins":
                _check_builtin_is_type(resolved, type_str)
            return resolved

        raise DeserializationError(f"Could not locate the type: {type_name} in the module: {module_name}")

    # No module prefix, check builtins and typing
    # Special cases for None / NoneType first: `getattr(builtins, "None")` returns the `None`
    # singleton (not a type), so these must be handled before the builtins type gate below.
    if type_str == "None":
        return None
    if type_str == "NoneType":
        return NoneType

    # Then check builtins
    if hasattr(builtins, type_str):
        resolved = getattr(builtins, type_str)
        # This bare-name path never consults the allowlist. A type annotation must resolve to an
        # actual type, so builtin functions like `eval`/`exec` are rejected while types pass.
        _check_builtin_is_type(resolved, type_str)
        return resolved

    # Then check typing
    if hasattr(typing, type_str):
        return getattr(typing, type_str)

    raise DeserializationError(f"Could not deserialize type: {type_str}")


def thread_safe_import(module_name: str) -> ModuleType:
    """
    Import a module in a thread-safe manner.

    Importing modules in a multi-threaded environment can lead to race conditions.
    This function ensures that the module is imported in a thread-safe manner without having impact
    on the performance of the import for single-threaded environments.

    :param module_name: the module to import
    """
    with _import_lock:
        return importlib.import_module(module_name)
