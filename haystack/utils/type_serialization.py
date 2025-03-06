# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import sys
import typing
from threading import Lock
from types import ModuleType
from typing import Any, get_args

from haystack import DeserializationError

_import_lock = Lock()


def serialize_type(target: Any) -> str:
    """
    Serializes a type or an instance to its string representation, including the module name.

    This function handles types, instances of types, and special typing objects.
    It assumes that non-typing objects will have a '__name__' attribute and raises
    an error if a type cannot be serialized.

    :param target:
        The object to serialize, can be an instance or a type.
    :return:
        The string representation of the type.
    :raises ValueError:
        If the type cannot be serialized.
    """
    name = getattr(target, "__name__", str(target))

    # Get module name
    module = inspect.getmodule(target)
    if module and hasattr(module, "__name__"):
        if module.__name__ == "builtins":
            # omit the module name for builtins, it just clutters the output
            # e.g. instead of 'builtins.str', we'll just return 'str'
            module_name = ""
        else:
            module_name = f"{module.__name__}"
    else:
        module_name = ""

    args = get_args(target)
    if args:
        args = ", ".join([serialize_type(a) for a in args if a is not type(None)])
        return f"{module_name}.{name}[{args}]" if module_name else f"{name}[{args}]"

    return f"{module_name}.{name}" if module_name else f"{name}"


def deserialize_type(type_str: str) -> Any:
    """
    Deserializes a type given its full import path as a string, including nested generic types.

    This function will dynamically import the module if it's not already imported
    and then retrieve the type object from it. It also handles nested generic types like
    `typing.List[typing.Dict[int, str]]`.

    :param type_str:
        The string representation of the type's full import path.
    :returns:
        The deserialized type object.
    :raises DeserializationError:
        If the type cannot be deserialized due to missing module or type.
    """

    type_mapping = {
        list: typing.List,
        dict: typing.Dict,
        set: typing.Set,
        tuple: typing.Tuple,
        frozenset: typing.FrozenSet,
    }

    def parse_generic_args(args_str):
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

    if "[" in type_str and type_str.endswith("]"):
        # Handle generics
        main_type_str, generics_str = type_str.split("[", 1)
        generics_str = generics_str[:-1]

        main_type = deserialize_type(main_type_str)
        generic_args = tuple(deserialize_type(arg) for arg in parse_generic_args(generics_str))

        # Reconstruct
        if sys.version_info >= (3, 9) or repr(main_type).startswith("typing."):
            return main_type[generic_args]
        else:
            return type_mapping[main_type][generic_args]  # type: ignore

    else:
        # Handle non-generics
        parts = type_str.split(".")
        module_name = ".".join(parts[:-1]) or "builtins"
        type_name = parts[-1]

        module = sys.modules.get(module_name)
        if not module:
            try:
                module = thread_safe_import(module_name)
            except ImportError as e:
                raise DeserializationError(f"Could not import the module: {module_name}") from e

        deserialized_type = getattr(module, type_name, None)
        if not deserialized_type:
            raise DeserializationError(f"Could not locate the type: {type_name} in the module: {module_name}")

        return deserialized_type


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
