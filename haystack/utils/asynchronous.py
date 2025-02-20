# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable


def is_callable_async_compatible(func: Callable) -> bool:
    """
    Returns if the given callable is usable inside a component's `run_async` method.

    :param callable:
        The callable to check.
    :returns:
        True if the callable is compatible, False otherwise.
    """
    return inspect.iscoroutinefunction(func)
