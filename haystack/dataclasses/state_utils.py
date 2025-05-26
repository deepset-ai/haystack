# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, List, TypeVar, Union

T = TypeVar("T")


def merge_lists(current: Union[List[T], T, None], new: Union[List[T], T]) -> List[T]:
    """
    Merges two values into a single list.

    Deprecated in favor of `haystack.components.agents.state.merge_lists`. It will be removed in Haystack 2.16.0.


    If either `current` or `new` is not already a list, it is converted into one.
    The function ensures that both inputs are treated as lists and concatenates them.

    If `current` is None, it is treated as an empty list.

    :param current: The existing value(s), either a single item or a list.
    :param new: The new value(s) to merge, either a single item or a list.
    :return: A list containing elements from both `current` and `new`.
    """

    warnings.warn(
        "`haystack.dataclasses.state_utils.merge_lists` is deprecated and will be removed in Haystack 2.16.0. "
        "Use `haystack.components.agents.state.merge_lists` instead.",
        DeprecationWarning,
    )
    current_list = [] if current is None else current if isinstance(current, list) else [current]
    new_list = new if isinstance(new, list) else [new]
    return current_list + new_list


def replace_values(current: Any, new: Any) -> Any:
    """
    Replace the `current` value with the `new` value.

    :param current: The existing value
    :param new: The new value to replace
    :return: The new value
    """

    warnings.warn(
        "`haystack.dataclasses.state_utils.replace_values` is deprecated and will be removed in Haystack 2.16.0. "
        "Use `haystack.components.agents.state.replace_values` instead.",
        DeprecationWarning,
    )
    return new
