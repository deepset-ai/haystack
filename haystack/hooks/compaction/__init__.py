# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "hooks": ["CompactionHook"],
    "sliding_window": ["SlidingWindowCompactor"],
    "tool_result_pruning": ["ToolResultPruningCompactor"],
    "types": ["Compactor"],
}

if TYPE_CHECKING:
    from .hooks import CompactionHook as CompactionHook
    from .sliding_window import SlidingWindowCompactor as SlidingWindowCompactor
    from .tool_result_pruning import ToolResultPruningCompactor as ToolResultPruningCompactor
    from .types import Compactor as Compactor
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
