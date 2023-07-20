# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, get_args, get_origin, Any

import logging


logger = logging.getLogger(__name__)


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
