# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "protocol": [
        "Hook",
        "HookPoint",
        "BEFORE_RUN",
        "BEFORE_LLM",
        "BEFORE_TOOL",
        "AFTER_TOOL",
        "ON_EXIT",
        "AFTER_RUN",
        "VALID_HOOK_POINTS",
    ],
    "from_function": ["FunctionHook", "hook"],
}

if TYPE_CHECKING:
    from .from_function import FunctionHook as FunctionHook
    from .from_function import hook as hook
    from .protocol import AFTER_RUN as AFTER_RUN
    from .protocol import AFTER_TOOL as AFTER_TOOL
    from .protocol import BEFORE_LLM as BEFORE_LLM
    from .protocol import BEFORE_RUN as BEFORE_RUN
    from .protocol import BEFORE_TOOL as BEFORE_TOOL
    from .protocol import ON_EXIT as ON_EXIT
    from .protocol import VALID_HOOK_POINTS as VALID_HOOK_POINTS
    from .protocol import Hook as Hook
    from .protocol import HookPoint as HookPoint
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
