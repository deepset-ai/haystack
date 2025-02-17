# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeVar, Union, get_args, get_origin

from haystack import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _old_types_are_compatible(sender, receiver):  # pylint: disable=too-many-return-statements
    """
    Checks whether the source type is equal or a subtype of the destination type. Used to validate pipeline connections.

    Note: this method has no pretense to perform proper type matching. It especially does not deal with aliasing of
    typing classes such as `List` or `Dict` to their runtime counterparts `list` and `dict`. It also does not deal well
    with "bare" types, so `List` is treated differently from `List[Any]`, even though they should be the same.

    Consider simplifying the typing of your components if you observe unexpected errors during component connection.
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

    sender_origin = get_origin(sender)
    receiver_origin = get_origin(receiver)

    if sender_origin is not Union and receiver_origin is Union:
        return any(_types_are_compatible(sender, union_arg) for union_arg in get_args(receiver))

    if not sender_origin or not receiver_origin or sender_origin != receiver_origin:
        return False

    sender_args = get_args(sender)
    receiver_args = get_args(receiver)
    if len(sender_args) > len(receiver_args):
        return False

    return all(_types_are_compatible(*args) for args in zip(sender_args, receiver_args))


def _types_are_compatible(type1, type2) -> bool:
    """
    Core type compatibility check implementing symmetric matching.

    :param type1: First unwrapped type to compare
    :param type2: Second unwrapped type to compare
    :return: True if types are compatible, False otherwise
    """
    # Handle Any type and direct equality
    if type1 is Any or type2 is Any or type1 == type2:
        return True

    # Added this line to handle classes and subclasses
    try:
        if issubclass(type2, type1) or issubclass(type1, type2):
            return True
    except TypeError:  # typing classes can't be used with issubclass, so we deal with them below
        pass

    type1_origin = get_origin(type1)
    type2_origin = get_origin(type2)

    # Handle Union types
    if type1_origin is Union or type2_origin is Union:
        return _check_union_compatibility(type1, type2, type1_origin, type2_origin)

    # Handle non-Union types
    return _check_non_union_compatibility(type1, type2, type1_origin, type2_origin)


def _check_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> bool:
    """Handle all Union type compatibility cases."""
    if type1_origin is Union and type2_origin is not Union:
        return any(_types_are_compatible(union_arg, type2) for union_arg in get_args(type1))
    if type2_origin is Union and type1_origin is not Union:
        return any(_types_are_compatible(type1, union_arg) for union_arg in get_args(type2))
    # Both are Union types
    return any(any(_types_are_compatible(arg1, arg2) for arg2 in get_args(type2)) for arg1 in get_args(type1))


def _check_non_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> bool:
    """Handle non-Union type compatibility cases."""
    # If no origin, compare types directly
    if not type1_origin and not type2_origin:
        return type1 == type2

    # Both must have origins and they must be equal
    if not (type1_origin and type2_origin and type1_origin == type2_origin):
        return False

    # Compare generic type arguments
    type1_args = get_args(type1)
    type2_args = get_args(type2)

    if len(type1_args) != len(type2_args):
        return False

    return all(_types_are_compatible(t1_arg, t2_arg) for t1_arg, t2_arg in zip(type1_args, type2_args))


def _type_name(type_):
    """
    Util methods to get a nice readable representation of a type.

    Handles Optional and Literal in a special way to make it more readable.
    """
    # Literal args are strings, so we wrap them in quotes to make it clear
    if isinstance(type_, str):
        return f"'{type_}'"

    name = getattr(type_, "__name__", str(type_))

    if name.startswith("typing."):
        name = name[7:]
    if "[" in name:
        name = name.split("[")[0]
    args = get_args(type_)
    if name == "Union" and type(None) in args and len(args) == 2:
        # Optional is technically a Union of type and None
        # but we want to display it as Optional
        name = "Optional"

    if args:
        args = ", ".join([_type_name(a) for a in args if a is not type(None)])
        return f"{name}[{args}]"

    return f"{name}"
