# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from functools import wraps


def _warn_on_inplace_mutation(cls: type) -> type:
    """
    Decorator that warns if the dataclass is mutated in-place.
    """
    initializing = set()

    original_init = getattr(cls, "__init__")
    original_setattr = getattr(cls, "__setattr__")

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
                f"'{type(self).__name__}' is deprecated. "
                f"Use `dataclasses.replace(instance, {name}=new_value)` instead. "
                "In-place modification of dataclass instances will be removed in a future version.",
                Warning,
                stacklevel=2,
            )
        return original_setattr(self, name, value)

    setattr(cls, "__init__", __init_track__)
    setattr(cls, "__setattr__", __setattr_warn__)
    return cls
