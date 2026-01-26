# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections.abc
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


def _types_are_compatible(sender: type | UnionType, receiver: type | UnionType, type_validation: bool = True) -> bool:
    """
    Determines if two types are compatible based on the specified validation mode.

    :param sender: The sender type.
    :param receiver: The receiver type.
    :param type_validation: Whether to perform strict type validation.
    :return: True if the types are compatible, False otherwise.
    """
    if type_validation:
        return _strict_types_are_compatible(sender, receiver) or _are_convertible(sender, receiver)
    return True


def _are_convertible(sender: Any, receiver: Any) -> bool:
    """
    Checks if the sender type can be implicitly converted to the receiver type.

    Supported conversions:
    - ChatMessage <-> str
    - T <-> List[T] (wrapping/unwrapping)
    - Combinations like List[ChatMessage] <-> str
    """
    # Strict Nullability: Optional[T] should not connect to T
    if _can_be_none(sender) and not _can_be_none(receiver):
        return False

    if _is_base_convertible(sender, receiver):
        return True

    s_origin, r_origin = _safe_get_origin(sender), _safe_get_origin(receiver)

    # Handle List[T] -> U (unwrapping)
    if s_origin in (list, list) and (s_args := get_args(sender)):
        if _strict_types_are_compatible(s_args[0], receiver) or _is_base_convertible(s_args[0], receiver):
            return True

    # Handle T -> List[U] (wrapping)
    if r_origin in (list, list) and (r_args := get_args(receiver)):
        if _strict_types_are_compatible(sender, r_args[0]) or _is_base_convertible(sender, r_args[0]):
            return True

    return False


def _convert_value(value: Any, sender_type: Any, receiver_type: Any) -> Any:
    """
    Converts a value from the sender type to the receiver type.
    """
    if _strict_types_are_compatible(sender_type, receiver_type):
        return value

    s_origin, r_origin = _safe_get_origin(sender_type), _safe_get_origin(receiver_type)

    # 1. Unwrapping: List[T] -> U
    if s_origin in (list, list) and (s_args := get_args(sender_type)):
        inner_s = s_args[0]
        if _strict_types_are_compatible(inner_s, receiver_type) or _is_base_convertible(inner_s, receiver_type):
            try:
                if value is not None and len(value) > 0:
                    return _convert_value(value[0], inner_s, receiver_type)
            except (TypeError, IndexError):
                pass

            # If we reach here, we couldn't get the first element
            if _can_be_none(receiver_type):
                return None
            raise ValueError(
                f"Cannot convert empty list (type {sender_type}) to non-optional type {receiver_type}. "
                "The list must contain at least one element."
            )

    # 2. Wrapping: T -> List[U]
    if r_origin in (list, list) and (r_args := get_args(receiver_type)):
        inner_r = r_args[0]
        if _strict_types_are_compatible(sender_type, inner_r) or _is_base_convertible(sender_type, inner_r):
            return [_convert_value(value, sender_type, inner_r)]

    if value is None:
        return None

    # 3. Base Conversion: ChatMessage <-> str
    converted = _convert_base(value, sender_type, receiver_type)

    if converted is None and not _can_be_none(receiver_type):
        raise ValueError(
            f"Cannot convert {sender_type} to non-optional type {receiver_type} because the result is None. "
            f"Source value: {value}"
        )
    return converted


def _is_chat_message(t: Any) -> bool:
    from haystack.dataclasses import ChatMessage

    return t == ChatMessage or (_safe_get_origin(t) is Union and any(arg == ChatMessage for arg in get_args(t)))


def _is_str(t: Any) -> bool:
    return t == str or (_safe_get_origin(t) is Union and any(arg == str for arg in get_args(t)))


def _is_base_convertible(s: Any, r: Any) -> bool:
    return (_is_chat_message(s) and _is_str(r)) or (_is_str(s) and _is_chat_message(r))


def _convert_base(value: Any, s_t: Any, r_t: Any) -> Any:
    from haystack.dataclasses import ChatMessage

    if _is_chat_message(s_t) and _is_str(r_t):
        return value.text if isinstance(value, ChatMessage) else str(value)
    if _is_str(s_t) and _is_chat_message(r_t):
        return ChatMessage.from_user(value) if isinstance(value, str) else value
    return value


def _can_be_none(t: Any) -> bool:
    """Checks if a type can be None (e.g., Optional[T] or Union[T, None])."""
    return t is NoneType or (_safe_get_origin(t) is Union and any(arg is NoneType for arg in get_args(t)))


def _safe_get_origin(_type: type[T]) -> Union[type[T], None]:
    """
    Safely retrieves the origin type of a generic alias or returns the type itself if it's a built-in.
    """
    origin = get_origin(_type) or (_type if isinstance(_type, type) else None)
    return Union if origin is UnionType else origin


def _strict_types_are_compatible(sender: Any, receiver: Any) -> bool:
    """
    Checks whether the sender type is equal to or a subtype of the receiver type under strict validation.
    """
    if sender == receiver or receiver is Any:
        return True
    if sender is Any:
        return False
    try:
        if issubclass(sender, receiver):
            return True
    except TypeError:
        pass

    sender_origin = _safe_get_origin(sender)
    receiver_origin = _safe_get_origin(receiver)

    if sender_origin is not Union and receiver_origin is Union:
        return any(_strict_types_are_compatible(sender, union_arg) for union_arg in get_args(receiver))

    if not (sender_origin and receiver_origin and sender_origin == receiver_origin):
        return False

    sender_args, receiver_args = get_args(sender), get_args(receiver)
    if sender_origin == receiver_origin == collections.abc.Callable:
        return _check_callable_compatibility(sender_args, receiver_args)

    if not sender_args and sender_origin:
        sender_args = (Any,)
    if not receiver_args and receiver_origin:
        receiver_args = (Any,) * (len(sender_args) if sender_args else 1)

    return len(sender_args) <= len(receiver_args) and all(
        _strict_types_are_compatible(*args) for args in zip(sender_args, receiver_args)
    )


def _check_callable_compatibility(sender_args: Any, receiver_args: Any) -> bool:
    """Helper function to check compatibility of Callable types"""
    if not receiver_args:
        return True
    if not sender_args:
        sender_args = ([Any] * len(receiver_args[0]), Any)
    if len(sender_args) != 2 or len(receiver_args) != 2:
        return False
    if not _strict_types_are_compatible(sender_args[1], receiver_args[1]):
        return False
    if len(sender_args[0]) != len(receiver_args[0]):
        return False
    return all(_strict_types_are_compatible(sender_args[0][i], receiver_args[0][i]) for i in range(len(sender_args[0])))


def _type_name(type_: Any) -> str:
    """Util methods to get a nice readable representation of a type."""
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
        name = "Optional"
    if args:
        args_str = ", ".join([_type_name(a) for a in args if a is not NoneType])
        return f"{name}[{args_str}]"
    return name
