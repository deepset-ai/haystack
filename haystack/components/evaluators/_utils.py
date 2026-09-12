# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def _make_hashable(value: Any) -> Any:
    """
    Convert nested lists, tuples, and dictionaries into hashable comparison values.

    Lists and tuples keep their order, while dictionaries ignore key insertion order, matching Python's equality
    semantics for these container types. Primitive values are returned unchanged.

    :param value: The value to convert.
    :returns: A hashable representation for values composed of supported containers and primitive values.
    """
    if isinstance(value, list):
        return ("list", tuple(_make_hashable(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_make_hashable(item) for item in value))
    if isinstance(value, dict):
        return ("dict", frozenset((key, _make_hashable(item)) for key, item in value.items()))
    return value
