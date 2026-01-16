# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for controlling progress bar behavior across Haystack components."""

import os

__all__ = ["HAYSTACK_PROGRESS_ENV", "get_progress_bar_setting"]

HAYSTACK_PROGRESS_ENV = "HAYSTACK_PROGRESS_BARS"
_DISABLE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_ENABLE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}


def get_progress_bar_setting(component_default: bool = True, env_var: str = HAYSTACK_PROGRESS_ENV) -> bool:
    """
    Determine whether to show progress bars based on component default and environment variable.

    This function allows users to globally control progress bar behavior across all Haystack
    components using the ``HAYSTACK_PROGRESS_BARS`` environment variable, while still
    allowing individual components to override this behavior.

    Priority order:
    1. Environment variable ``HAYSTACK_PROGRESS_BARS`` (if set and valid)
    2. Component's ``progress_bar`` parameter (``component_default``)

    :param component_default: The component's default progress_bar value.
        This is used when the environment variable is not set or has an invalid value.
    :param env_var: Name of the environment variable to check.
        Defaults to ``HAYSTACK_PROGRESS_BARS``.
    :return: ``True`` if progress bars should be shown, ``False`` otherwise.

    **Environment variable values:**

    To **disable** progress bars, set one of:
    - ``"0"``, ``"false"``, ``"no"``, ``"off"``, ``"disable"``, ``"disabled"``

    To **enable** progress bars, set one of:
    - ``"1"``, ``"true"``, ``"yes"``, ``"on"``, ``"enable"``, ``"enabled"``

    If the environment variable is unset or contains an invalid value,
    the ``component_default`` value is used.

    **Examples:**

    Disable all progress bars globally:

    .. code-block:: bash

        export HAYSTACK_PROGRESS_BARS=0

    .. code-block:: python

        from haystack.utils import get_progress_bar_setting

        class MyComponent:
            def __init__(self, progress_bar: bool = True):
                self.progress_bar = get_progress_bar_setting(progress_bar)

    Enable all progress bars globally (even if component default is False):

    .. code-block:: bash

        export HAYSTACK_PROGRESS_BARS=1

    No environment variable set - use component default:

    .. code-block:: python

        assert get_progress_bar_setting(True) is True
        assert get_progress_bar_setting(False) is False
    """
    env_value = os.getenv(env_var, "").lower()

    if env_value in _DISABLE_VALUES:
        return False
    if env_value in _ENABLE_VALUES:
        return True

    # Environment variable not set or has invalid value -> use component default
    return component_default
