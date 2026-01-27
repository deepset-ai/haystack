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
    - T -> List[T] (wrapping)
    - List[T] -> T (unwrapping - RESTRICTED to ChatMessage and str)
    - Combinations like List[ChatMessage] <-> str

    :param sender: The sender type.
    :param receiver: The receiver type.
    :return: True if the types are convertible, False otherwise.
    """
    # Strict Nullability: Optional[T] should not connect to T
    if _can_be_none(sender) and not _can_be_none(receiver):
        return False

    # Base: ChatMessage <-> str
    if _is_base_convertible(sender, receiver):
        return True

    s_origin, r_origin = _safe_get_origin(sender), _safe_get_origin(receiver)

    # Unwrap: List[T] -> U (Restricted to text/chat receivers)
    if s_origin is list and (s_args := get_args(sender)):
        inner_s = s_args[0]
        if _is_text_or_chat(receiver) and (_strict_types_are_compatible(inner_s, receiver) or _is_base_convertible(inner_s, receiver)):
            return True

    # Wrap: T -> List[U]
    if r_origin is list and (r_args := get_args(receiver)):
        inner_r = r_args[0]
        if _strict_types_are_compatible(sender, inner_r) or _is_base_convertible(sender, inner_r):
            return True

    return False


def _convert_value(value: Any, sender_type: Any, receiver_type: Any) -> Any:
    """
    Converts a value from the sender type to the receiver type at runtime.

    :param value: The value to convert.
    :param sender_type: The type of the value.
    :param receiver_type: The type to convert to.
    :return: The converted value.
    """
    if _strict_types_are_compatible(sender_type, receiver_type):
        return value

    s_origin, r_origin = _safe_get_origin(sender_type), _safe_get_origin(receiver_type)

    # 1. Unwrap: List[T] -> U (only for text/chat)
    if s_origin is list and _is_text_or_chat(receiver_type):
        try:
            if value and len(value) > 0:
                # Recursively convert the first element to the receiver type
                return _convert_value(value[0], get_args(sender_type)[0], receiver_type)
        except (TypeError, IndexError):
            pass

        if _can_be_none(receiver_type):
            return None
        raise ValueError(f"Cannot convert empty list to non-optional {receiver_type}.")

    # 2. Wrap: T -> List[U]
    if r_origin is list:
        # Recursively convert the value to the inner type of the list
        return [_convert_value(value, sender_type, get_args(receiver_type)[0])]

    # 3. Base: ChatMessage <-> str
    if value is None:
        if _can_be_none(receiver_type):
            return None
        raise ValueError(f"Cannot convert None to non-optional type {receiver_type}.")

    converted = _convert_base(value, sender_type, receiver_type)
    if converted is None and not _can_be_none(receiver_type):
        raise ValueError(f"Conversion of {value} to {receiver_type} resulted in None.")
    return converted


def _is_text_or_chat(t: Any) -> bool:
    """Returns True if t is str or ChatMessage (including Optional/Union)."""
    from haystack.dataclasses import ChatMessage
    args = get_args(t) if _safe_get_origin(t) is Union else (t,)
    return any(arg in (str, ChatMessage) for arg in args)


def _is_base_convertible(s: Any, r: Any) -> bool:
    """Returns True if s and r are a (str, ChatMessage) pair in any order."""
    from haystack.dataclasses import ChatMessage
    s_is_chat, r_is_chat = _is_type(s, ChatMessage), _is_type(r, ChatMessage)
    s_is_str, r_is_str = _is_type(s, str), _is_type(r, str)
    return (s_is_chat and r_is_str) or (s_is_str and r_is_chat)


def _is_type(t: Any, target: type) -> bool:
    """Checks if target is present in t (handles Union/Optional)."""
    args = get_args(t) if _safe_get_origin(t) is Union else (t,)
    return any(arg == target for arg in args)


def _convert_base(value: Any, s_t: Any, r_t: Any) -> Any:
    """Performs the actual transformation between str and ChatMessage."""
    from haystack.dataclasses import ChatMessage
    if _is_type(s_t, ChatMessage) and _is_type(r_t, str):
        return value.text if isinstance(value, ChatMessage) else str(value)
    if _is_type(s_t, str) and _is_type(r_t, ChatMessage):
        return ChatMessage.from_user(value) if isinstance(value, str) else value
    return value


def _can_be_none(t: Any) -> bool:
    """Checks if a type can be None (e.g., Optional[T] or Union[T, None])."""
    return t is NoneType or (_safe_get_origin(t) is Union and any(arg is NoneType for arg in get_args(t)))


def _safe_get_origin(_type: type[T]) -> Union[type[T], None]:
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


def _strict_types_are_compatible(sender: Any, receiver: Any) -> bool:
    """
    Checks whether the sender type is equal to or a subtype of the receiver type under strict validation.

    Note: this method has no pretense to perform proper type matching. It especially does not deal with aliasing of
    typing classes such as `List` or `Dict` to their runtime counterparts `list` and `dict`. It also does not deal well
    with "bare" types, so `List` is treated differently from `List[Any]`, even though they should be the same.
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
    sender_args, receiver_args = get_args(sender), get_args(receiver)

    # Handle Callable types
    if sender_origin == receiver_origin == collections.abc.Callable:
        return _check_callable_compatibility(sender_args, receiver_args)

    # Handle bare types
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
    return name
