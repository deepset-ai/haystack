# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


def _warn_on_inplace_mutation(cls: T) -> T:
    """
    Decorator that warns if the dataclass is mutated in-place.
    """
    initializing = set()

    # mypy requires using getattr/setattr for dunder access, but ruff prefers
    # direct attribute access. We silence mypy here in favor of the more explicit syntax.
    original_init = cls.__init__  # type: ignore[misc]
    original_setattr = cls.__setattr__

    @wraps(original_init)
    def __init_track__(self, *args, **kwargs):
        # We don't raise warnings during initialization, i.e. during the first call to __init__ and __post_init__.
        initializing.add(id(self))
        try:
            return original_init(self, *args, **kwargs)
        finally:
            initializing.discard(id(self))

    @wraps(original_setattr)
    def __setattr_warn__(self, name, value):
        # We raise warnings if the dataclass is mutated in-place after initialization.
        if (
            id(self) not in initializing
            and name in getattr(self, "__dataclass_fields__", {})
            and name in getattr(self, "__dict__", {})
        ):
            # We raise a warning if the attribute is a dataclass field and a dictionary key.
            warnings.warn(
                f"Mutating attribute '{name}' on an instance of "
                f"'{type(self).__name__}' can lead to unexpected behavior by affecting other parts of the pipeline "
                "that use the same dataclass instance. "
                f"Use `dataclasses.replace(instance, {name}=new_value)` instead. "
                "See https://docs.haystack.deepset.ai/docs/custom-components#requirements for details.",
                Warning,
                stacklevel=2,
            )
        # mypy infers original_setattr as bound to the type, expecting (str, Any), we call the unbound form
        return original_setattr(self, name, value)  # type: ignore[call-arg]

    # mypy considers direct dunder access on a class unsound, ruff prefers direct access
    cls.__init__ = __init_track__  # type: ignore[misc]
    # mypy does not allow assigning to a method, ruff prefers direct access
    cls.__setattr__ = __setattr_warn__  # type: ignore[method-assign, assignment]
    return cls
