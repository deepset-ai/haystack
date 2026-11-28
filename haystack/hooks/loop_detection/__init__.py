# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"detector": ["LoopDetectionResult", "LoopKind"], "hooks": ["ToolLoopDetectionHook"]}

if TYPE_CHECKING:
    from .detector import LoopDetectionResult as LoopDetectionResult
    from .detector import LoopKind as LoopKind
    from .hooks import ToolLoopDetectionHook as ToolLoopDetectionHook
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
