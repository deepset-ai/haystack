# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os


def _get_progress_bar_setting(default: bool) -> bool:
    """Resolve the global override at execution time without changing component configuration."""
    value = os.getenv("HAYSTACK_PROGRESS_BARS", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default
