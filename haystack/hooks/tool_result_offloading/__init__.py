# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "hooks": ["ToolResultOffloadHook", "RESULT_STORE_CONTEXT_KEY"],
    "policies": ["AlwaysOffload", "NeverOffload", "OffloadOverChars"],
    "stores": ["FileSystemToolResultStore"],
    "types": ["OffloadPolicy", "ToolResultStore"],
}

if TYPE_CHECKING:
    from .hooks import RESULT_STORE_CONTEXT_KEY as RESULT_STORE_CONTEXT_KEY
    from .hooks import ToolResultOffloadHook as ToolResultOffloadHook
    from .policies import AlwaysOffload as AlwaysOffload
    from .policies import NeverOffload as NeverOffload
    from .policies import OffloadOverChars as OffloadOverChars
    from .stores import FileSystemToolResultStore as FileSystemToolResultStore
    from .types import OffloadPolicy as OffloadPolicy
    from .types import ToolResultStore as ToolResultStore
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
