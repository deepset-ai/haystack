# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings
from typing import Any, TypeVar

T = TypeVar("T")


def _experimental(cls: type[T]) -> type[T]:
    """
    Class decorator that marks a Haystack component as experimental.

    Components decorated with @experimental are subject to breaking changes
    or removal in future releases without prior deprecation notice.

    ## Usage example

        @_experimental
        @component
        class MyComponent:
            ...
    """
    original_init: Any = cls.__init__

    @functools.wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            f"'{cls.__name__}' is an experimental component and may change or be removed "
            "in future releases without prior deprecation notice. ",
            ExperimentalWarning,
            stacklevel=2,
        )
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    cls.__experimental__ = True
    return cls


class ExperimentalWarning(UserWarning):
    """Warning emitted when an experimental Haystack component is instantiated."""
