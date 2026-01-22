# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"state": ["State", "merge_lists", "replace_values"]}

if TYPE_CHECKING:
    from .state import State as State
    from .state_utils import merge_lists as merge_lists
    from .state_utils import replace_values as replace_values

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
