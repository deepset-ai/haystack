# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "protocol": ["Hook", "HookEvent", "BEFORE_LLM", "BEFORE_TOOL", "ON_EXIT", "VALID_HOOK_EVENTS"],
    "from_function": ["FunctionHook", "hook"],
}

if TYPE_CHECKING:
    from .from_function import FunctionHook as FunctionHook
    from .from_function import hook as hook
    from .protocol import BEFORE_LLM as BEFORE_LLM
    from .protocol import BEFORE_TOOL as BEFORE_TOOL
    from .protocol import ON_EXIT as ON_EXIT
    from .protocol import VALID_HOOK_EVENTS as VALID_HOOK_EVENTS
    from .protocol import Hook as Hook
    from .protocol import HookEvent as HookEvent
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
