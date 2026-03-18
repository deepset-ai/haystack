# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "gnap_task_creator": ["GNAPTaskCreator"],
    "gnap_result_reader": ["GNAPResultReader"],
}

if TYPE_CHECKING:
    from .gnap_result_reader import GNAPResultReader as GNAPResultReader
    from .gnap_task_creator import GNAPTaskCreator as GNAPTaskCreator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
