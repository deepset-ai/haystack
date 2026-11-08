# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "approximate_counter": ["ApproximateTokenCounter"],
    "openai_counter": ["OpenAITokenCounter"],
    "types": ["TokenCounter"],
    "tiktoken_counter": ["TiktokenCounter"],
}

if TYPE_CHECKING:
    from .approximate_counter import ApproximateTokenCounter as ApproximateTokenCounter
    from .openai_counter import OpenAITokenCounter as OpenAITokenCounter
    from .tiktoken_counter import TiktokenCounter as TiktokenCounter
    from .types import TokenCounter as TokenCounter
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
