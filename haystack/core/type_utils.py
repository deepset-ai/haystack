# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections.abc
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from haystack.dataclasses import ChatMessage


def _types_are_compatible(
    sender: type | UnionType, receiver: type | UnionType, type_validation: bool = True
) -> tuple[bool, bool]:
    """
    Determines if two types are compatible based on the specified validation mode.

    :param sender: The sender type.
    :param receiver: The receiver type.
    :param type_validation: Whether to perform strict type validation.
    :return: A tuple where the first element is True if the types are compatible, and the second
             element is True if they are strictly compatible.
    """
    if not type_validation:
        return True, True

    if _strict_types_are_compatible(sender, receiver):
        return True, True

    if _types_are_convertible(sender, receiver):
        return True, False

    return False, False


def _safe_get_origin(_type: type | UnionType) -> type | None:
    """
    Safely retrieves the origin type of a generic alias or returns the type itself if it's a built-in.

    This function extends the behavior of `typing.get_origin()` by also handling plain built-in types
    like `list`, `dict`, etc., which `get_origin()` would normally return `None` for.

    :param _type: A type or generic alias (e.g., `list`, `list[int]`, `dict[str, int]`).

    :returns: The origin type (e.g., `list`, `dict`), or `None` if the input is not a type.
    """
    origin = get_origin(_type) or (_type if isinstance(_type, type) else None)
    # We want to treat typing.Union and UnionType as the same for compatibility checks.
    # So we convert UnionType to Union if it is detected.
    if origin is UnionType:
        origin = Union
    return origin


def _contains_type(container: Any, target: Any) -> bool:
    """Checks if the container type includes the target type"""
    if container is target:
        return True
    return _safe_get_origin(container) is Union and target in get_args(container)


def _types_are_convertible(sender: Any, receiver: Any) -> bool:
    """
    Checks whether the sender type is convertible to the receiver type.

    ChatMessage is convertible to str and vice versa.
    """
    # Optional[T] must not connect to T
    if _contains_type(sender, NoneType) and not _contains_type(receiver, NoneType):
        return False

    if _contains_type(receiver, sender):
        return True

    if _contains_type(sender, ChatMessage) and _contains_type(receiver, str):
        return True

    return _contains_type(sender, str) and _contains_type(receiver, ChatMessage)


def _convert_value(value: Any, sender_type: Any, receiver_type: Any) -> Any:
    """
    Converts a value from the sender type to the receiver type.

    :param value: The value to convert.
    :param sender_type: The sender type.
    :param receiver_type: The receiver type.
    :return: The converted value.
    """
    if _contains_type(sender_type, ChatMessage) and _contains_type(receiver_type, str):
        if value.text is None:
            msg = "Cannot convert `ChatMessage` to `str` because it has no text. "
            raise ValueError(msg)
        return value.text

    if _contains_type(sender_type, str) and _contains_type(receiver_type, ChatMessage):
        return ChatMessage.from_user(value)

    return value


def _strict_types_are_compatible(sender: Any, receiver: Any) -> bool:  # pylint: disable=too-many-return-statements
    """
    Checks whether the sender type is equal to or a subtype of the receiver type under strict validation.

    Note: this method has no pretense to perform complete type matching.
    Consider simplifying the typing of your components if you observe unexpected errors during component connection.

    :param sender: The sender type.
    :param receiver: The receiver type.
    :return: True if the sender type is strictly compatible with the receiver type, False otherwise.
    """
    if sender == receiver or receiver is Any:
        return True

    if sender is Any:
        return False

    try:
        if issubclass(sender, receiver):
            return True
    except TypeError:  # typing classes can't be used with issubclass, so we deal with them below
        pass

    sender_origin = _safe_get_origin(sender)
    receiver_origin = _safe_get_origin(receiver)

    if sender_origin is not Union and receiver_origin is Union:
        return any(_strict_types_are_compatible(sender, union_arg) for union_arg in get_args(receiver))

    # Both must have origins and they must be equal
    if not (sender_origin and receiver_origin and sender_origin == receiver_origin):
        return False

    # Compare generic type arguments
    sender_args = get_args(sender)
    receiver_args = get_args(receiver)

    # Handle Callable types
    if sender_origin == receiver_origin == collections.abc.Callable:
        return _check_callable_compatibility(sender_args, receiver_args)

    # Handle bare types
    if not sender_args and sender_origin:
        sender_args = (Any,)
    if not receiver_args and receiver_origin:
        receiver_args = (Any,) * (len(sender_args) if sender_args else 1)

    return not (len(sender_args) > len(receiver_args)) and all(
        _strict_types_are_compatible(*args) for args in zip(sender_args, receiver_args)
    )


def _check_callable_compatibility(sender_args, receiver_args):
    """Helper function to check compatibility of Callable types"""
    if not receiver_args:
        return True
    if not sender_args:
        sender_args = ([Any] * len(receiver_args[0]), Any)
    # Standard Callable has two elements in args: argument list and return type
    if len(sender_args) != 2 or len(receiver_args) != 2:
        return False
    # Return types must be compatible
    if not _strict_types_are_compatible(sender_args[1], receiver_args[1]):
        return False
    # Input Arguments must be of same length
    if len(sender_args[0]) != len(receiver_args[0]):
        return False
    return all(_strict_types_are_compatible(sender_args[0][i], receiver_args[0][i]) for i in range(len(sender_args[0])))


def _type_name(type_: Any) -> str:
    """
    Util methods to get a nice readable representation of a type.

    Handles Optional and Literal in a special way to make it more readable.
    """
    # Literal args are strings, so we wrap them in quotes to make it clear
    if isinstance(type_, str):
        return f"'{type_}'"

    if type_ is NoneType:
        return "None"

    args = get_args(type_)

    if isinstance(type_, UnionType):
        return " | ".join([_type_name(a) for a in args])

    name = getattr(type_, "__name__", str(type_))
    if name.startswith("typing."):
        name = name[7:]
    if "[" in name:
        name = name.split("[")[0]

    if name == "Union" and NoneType in args and len(args) == 2:
        # Optional is technically a Union of type and None
        # but we want to display it as Optional
        name = "Optional"

    if args:
        args_str = ", ".join([_type_name(a) for a in args if a is not NoneType])
        return f"{name}[{args_str}]"

    return f"{name}"
