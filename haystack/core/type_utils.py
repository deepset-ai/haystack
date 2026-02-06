# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections.abc
from types import NoneType, UnionType
from typing import Any, Callable, Union, get_args, get_origin

from haystack.dataclasses import ChatMessage


def _chat_message_to_str(value: Any) -> str:
    if value.text is None:
        raise ValueError("Cannot convert `ChatMessage` to `str` because it has no text. ")
    return value.text


_CONVERSION_STRATEGIES: dict[str, Callable[[Any], Any]] = {
    "chat_message_to_str": _chat_message_to_str,
    "str_to_chat_message": ChatMessage.from_user,
    "wrap": lambda v: [v],
    "wrap_chat_message_to_str": lambda v: [_chat_message_to_str(v)],
    "wrap_str_to_chat_message": lambda v: [ChatMessage.from_user(v)],
}


def _types_are_compatible(sender: Any, receiver: Any, type_validation: bool = True) -> tuple[bool, str | None]:
    """
    Determines if two types are compatible based on the specified validation mode.

    :param sender: The sender type.
    :param receiver: The receiver type.
    :param type_validation: Whether to perform strict type validation.
    :return: A tuple where the first element is True if the types are compatible, and the second
             element is the conversion strategy name (or None if strictly compatible or type validation is disabled).
    """
    if not type_validation:
        return True, None

    if _strict_types_are_compatible(sender, receiver):
        return True, None

    strategy = _get_conversion_strategy(sender, receiver)
    if strategy:
        return True, strategy

    return False, None


def _safe_get_origin(_type: Any) -> type | None:
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
    if container == target:
        return True
    return _safe_get_origin(container) is Union and target in get_args(container)


def _get_conversion_strategy(sender: Any, receiver: Any) -> str | None:  # pylint: disable=too-many-return-statements
    """
    Returns the name of the conversion strategy to use for the given sender and receiver types.
    """
    # Optional[T] must not connect to T
    if _contains_type(sender, NoneType) and not _contains_type(receiver, NoneType):
        return None

    # Base: ChatMessage -> str
    if _contains_type(sender, ChatMessage) and _contains_type(receiver, str):
        return "chat_message_to_str"

    # Base: str -> ChatMessage
    if _contains_type(sender, str) and _contains_type(receiver, ChatMessage):
        return "str_to_chat_message"

    # Wrap: T -> List[T]
    if _safe_get_origin(receiver) is list and (args := get_args(receiver)):
        inner = args[0]
        if _strict_types_are_compatible(sender, inner):
            return "wrap"
        if _contains_type(sender, ChatMessage) and _contains_type(inner, str):
            return "wrap_chat_message_to_str"
        if _contains_type(sender, str) and _contains_type(inner, ChatMessage):
            return "wrap_str_to_chat_message"

    return None


def _convert_value(value: Any, strategy: str) -> Any:
    """
    Converts a value from the sender type to the receiver type using a strategy.
    """
    return _CONVERSION_STRATEGIES[strategy](value)


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
