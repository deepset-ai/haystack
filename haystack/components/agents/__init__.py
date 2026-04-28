# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"agent": ["Agent"], "state": ["State"], "tool_invoker": ["ToolInvoker"]}

if TYPE_CHECKING:
    from .agent import Agent as Agent
    from .state import State as State
    from .tool_invoker import ToolInvoker as ToolInvoker

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
