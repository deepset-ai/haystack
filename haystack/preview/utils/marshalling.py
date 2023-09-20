from typing import Type
import builtins
import sys

from haystack.preview import DeserializationError


def marshal_type(type_: Type) -> str:
    """
    Given a type, return a string representation that can be unmarshalled.

    :param type_: The type.
    :return: Its string representation.
    """
    module = type_.__module__
    if module == "builtins":
        return type_.__name__
    return f"{module}.{type_.__name__}"


def unmarshal_type(type_name: str) -> Type:
    """
    Given the string representation of a type, return the type itself.

    :param type_name: The string representation of the type.
    :return: The type itself.
    """
    if "." not in type_name:
        type_ = getattr(builtins, type_name, None)
        if not type_:
            raise DeserializationError(f"Could not locate builtin called '{type_name}'")
        return type_

    parts = type_name.split(".")
    module_name = ".".join(parts[:-1])
    type_name = parts[-1]

    module = sys.modules.get(module_name, None)
    if not module:
        raise DeserializationError(f"Could not locate the module '{module_name}'")

    type_ = getattr(module, type_name, None)
    if not type_:
        raise DeserializationError(f"Could not locate the type '{type_name}'")

    return type_
