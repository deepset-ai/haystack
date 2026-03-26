# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "state": ["State", "merge_lists", "replace_values"],
    "state_tools": ["LsStateTool", "ReadStateTool", "WriteStateTool", "StateToolset"],
}

if TYPE_CHECKING:
    from .state import State as State
    from .state_tools import LsStateTool as LsStateTool
    from .state_tools import ReadStateTool as ReadStateTool
    from .state_tools import StateToolset as StateToolset
    from .state_tools import WriteStateTool as WriteStateTool
    from .state_utils import merge_lists as merge_lists
    from .state_utils import replace_values as replace_values

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
