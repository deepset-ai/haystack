# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union, get_args, get_origin

from haystack import logging

logger = logging.getLogger(__name__)


def _is_optional(type_: type) -> bool:
    """
    Utility method that returns whether a type is Optional.
    """
    return get_origin(type_) is Union and type(None) in get_args(type_)


def _types_are_compatible(sender, receiver):  # pylint: disable=too-many-return-statements
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
