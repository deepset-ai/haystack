# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "answer_builder": ["AnswerBuilder"],
    "chat_prompt_builder": ["ChatPromptBuilder"],
    "prompt_builder": ["PromptBuilder"],
}

if TYPE_CHECKING:
    from .answer_builder import AnswerBuilder
    from .chat_prompt_builder import ChatPromptBuilder
    from .prompt_builder import PromptBuilder

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
