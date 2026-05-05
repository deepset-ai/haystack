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
    # getattr/setattr are intentional here: direct attribute access (cls.__init__, cls.__init__ = ...)
    # triggers mypy [misc] and [attr-defined] errors because T is an unbound TypeVar.
    # noqa comments suppress ruff B009/B010 which would auto-revert these back to direct access.
    original_init: Any = getattr(cls, "__init__")  # noqa: B009

    @functools.wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            f"'{cls.__name__}' is an experimental component and may change or be removed "
            "in future releases without prior deprecation notice. ",
            ExperimentalWarning,
            stacklevel=2,
        )
        original_init(self, *args, **kwargs)

    setattr(cls, "__init__", new_init)  # noqa: B010
    setattr(cls, "__experimental__", True)  # noqa: B010
    return cls


class ExperimentalWarning(UserWarning):
    """Warning emitted when an experimental Haystack component is instantiated."""
