# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "answer_joiner": ["AnswerJoiner"],
    "branch": ["BranchJoiner"],
    "document_joiner": ["DocumentJoiner"],
    "list_joiner": ["ListJoiner"],
    "string_joiner": ["StringJoiner"],
}

if TYPE_CHECKING:
    from .answer_joiner import AnswerJoiner as AnswerJoiner
    from .branch import BranchJoiner as BranchJoiner
    from .document_joiner import DocumentJoiner as DocumentJoiner
    from .list_joiner import ListJoiner as ListJoiner
    from .string_joiner import StringJoiner as StringJoiner

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
