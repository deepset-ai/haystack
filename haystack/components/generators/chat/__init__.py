# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "openai": ["OpenAIChatGenerator"],
    "azure": ["AzureOpenAIChatGenerator"],
    "hugging_face_local": ["HuggingFaceLocalChatGenerator"],
    "hugging_face_api": ["HuggingFaceAPIChatGenerator"],
}

if TYPE_CHECKING:
    from .azure import AzureOpenAIChatGenerator
    from .hugging_face_api import HuggingFaceAPIChatGenerator
    from .hugging_face_local import HuggingFaceLocalChatGenerator
    from .openai import OpenAIChatGenerator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
