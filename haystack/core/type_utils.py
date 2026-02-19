# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections.abc
from enum import Enum
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

from haystack.dataclasses import ChatMessage


class ConversionStrategy(Enum):
    """
    Strategies for converting values between compatible types in pipeline connections.
    """

    CHAT_MESSAGE_TO_STR = "chat_message_to_str"
    STR_TO_CHAT_MESSAGE = "str_to_chat_message"
    WRAP = "wrap"
    WRAP_CHAT_MESSAGE_TO_STR = "wrap_chat_message_to_str"
    WRAP_STR_TO_CHAT_MESSAGE = "wrap_str_to_chat_message"
    UNWRAP = "unwrap"
    UNWRAP_STR_TO_CHAT_MESSAGE = "unwrap_str_to_chat_message"
    UNWRAP_CHAT_MESSAGE_TO_STR = "unwrap_chat_message_to_str"


ConversionStrategyType = ConversionStrategy | None


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
    if container == target:
        return True
    return _safe_get_origin(container) is Union and target in get_args(container)


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


def _get_conversion_strategy(sender: Any, receiver: Any) -> ConversionStrategyType:  # pylint: disable=too-many-return-statements # noqa: PLR0911
    """
    Determines whether a conversion exists from sender to receiver.

    :param sender: The sender type.
    :param receiver: The receiver type.

    :returns: The ConversionStrategy if conversion is required and supported, otherwise None.
    """
    # If sender is a Union, it's only compatible if ALL its types are compatible with the same strategy
    if _safe_get_origin(sender) is Union:
        strategies = {_get_conversion_strategy(arg, receiver) for arg in get_args(sender)}
        if len(strategies) == 1:
            return strategies.pop()
        return None

    # If receiver is a Union, it's compatible if ANY of its types are compatible.
    # We prefer strategies that don't require type conversion if possible.
    if _safe_get_origin(receiver) is Union:
        strategies = {_get_conversion_strategy(sender, arg) for arg in get_args(receiver)} - {None}
        for preferred in (ConversionStrategy.WRAP, ConversionStrategy.UNWRAP):
            if preferred in strategies:
                return preferred
        return strategies.pop() if strategies else None

    # ChatMessage -> str
    if sender is ChatMessage and receiver is str:
        return ConversionStrategy.CHAT_MESSAGE_TO_STR

    # str -> ChatMessage
    if sender is str and receiver is ChatMessage:
        return ConversionStrategy.STR_TO_CHAT_MESSAGE

    # Wrap: T -> List[T]
    if _safe_get_origin(receiver) is list and (args := get_args(receiver)):
        inner = args[0]
        if _strict_types_are_compatible(sender, inner):
            return ConversionStrategy.WRAP
        # Wrap + conversion
        if _contains_type(sender, ChatMessage) and _contains_type(inner, str):
            return ConversionStrategy.WRAP_CHAT_MESSAGE_TO_STR
        if _contains_type(sender, str) and _contains_type(inner, ChatMessage):
            return ConversionStrategy.WRAP_STR_TO_CHAT_MESSAGE

    # Unwrap: List[T] -> T - for str and ChatMessage only
    if _safe_get_origin(sender) is list and (args := get_args(sender)):
        inner = args[0]
        # Guard against multi-level unwrap (e.g. list[list[str]] -> list[str])
        if _safe_get_origin(receiver) is not list and _strict_types_are_compatible(inner, receiver):
            return ConversionStrategy.UNWRAP
        # Unwrap + conversion: we need to check if all possible types of the sender list are compatible with the
        # receiver. We do this by recursively calling _get_conversion_strategy on the inner element type.
        inner_strategy = _get_conversion_strategy(inner, receiver)
        if inner_strategy == ConversionStrategy.STR_TO_CHAT_MESSAGE:
            return ConversionStrategy.UNWRAP_STR_TO_CHAT_MESSAGE
        if inner_strategy == ConversionStrategy.CHAT_MESSAGE_TO_STR:
            return ConversionStrategy.UNWRAP_CHAT_MESSAGE_TO_STR

    return None


def _types_are_compatible(
    sender: Any, receiver: Any, type_validation: bool = True
) -> tuple[bool, ConversionStrategyType]:
    """
    Determines whether two types are compatible, optionally allowing conversion.

    :param sender: The sender type.
    :param receiver: The receiver type.
    :param type_validation: If False, all types are considered compatible.

    :returns: A tuple of (is_compatible, conversion_strategy) where:
        - is_compatible is True if the types are strictly compatible or can be converted.
        - conversion_strategy is a ConversionStrategy if conversion is required, otherwise None
          (including when types are strictly compatible, incompatible, or type validation is disabled).
    """
    if not type_validation:
        return True, None

    if _strict_types_are_compatible(sender, receiver):
        return True, None

    conversion_strategy = _get_conversion_strategy(sender, receiver)
    if conversion_strategy:
        return True, conversion_strategy

    return False, None


def _chat_message_to_str(value: Any) -> str:
    if value.text is None:
        raise ValueError("Cannot convert `ChatMessage` to `str` because it has no text. ")
    return value.text


def _get_first_item(value: list[Any]) -> Any:
    if not value:
        raise ValueError("Cannot get first item of an empty list. ")
    return value[0]


def _convert_value(value: Any, conversion_strategy: ConversionStrategy) -> Any:  # pylint: disable=too-many-return-statements # noqa: PLR0911
    """
    Converts a value using the specified conversion strategy.
    """
    match conversion_strategy:
        case ConversionStrategy.CHAT_MESSAGE_TO_STR:
            return _chat_message_to_str(value)
        case ConversionStrategy.STR_TO_CHAT_MESSAGE:
            return ChatMessage.from_user(value)
        case ConversionStrategy.WRAP:
            return [value]
        case ConversionStrategy.WRAP_CHAT_MESSAGE_TO_STR:
            return [_chat_message_to_str(value)]
        case ConversionStrategy.WRAP_STR_TO_CHAT_MESSAGE:
            return [ChatMessage.from_user(value)]
        case ConversionStrategy.UNWRAP:
            return _get_first_item(value)
        case ConversionStrategy.UNWRAP_STR_TO_CHAT_MESSAGE:
            return ChatMessage.from_user(_get_first_item(value))
        case ConversionStrategy.UNWRAP_CHAT_MESSAGE_TO_STR:
            return _chat_message_to_str(_get_first_item(value))
