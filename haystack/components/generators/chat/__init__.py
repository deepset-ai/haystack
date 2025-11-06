# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "openai": ["OpenAIChatGenerator"],
    "openai_responses": ["OpenAIResponsesChatGenerator"],
    "azure": ["AzureOpenAIChatGenerator"],
    "azure_responses": ["AzureOpenAIResponsesChatGenerator"],
    "hugging_face_local": ["HuggingFaceLocalChatGenerator"],
    "hugging_face_api": ["HuggingFaceAPIChatGenerator"],
    "fallback": ["FallbackChatGenerator"],
}

if TYPE_CHECKING:
    from .azure import AzureOpenAIChatGenerator as AzureOpenAIChatGenerator
    from .azure_responses import AzureOpenAIResponsesChatGenerator as AzureOpenAIResponsesChatGenerator
    from .fallback import FallbackChatGenerator as FallbackChatGenerator
    from .hugging_face_api import HuggingFaceAPIChatGenerator as HuggingFaceAPIChatGenerator
    from .hugging_face_local import HuggingFaceLocalChatGenerator as HuggingFaceLocalChatGenerator
    from .openai import OpenAIChatGenerator as OpenAIChatGenerator
    from .openai_responses import OpenAIResponsesChatGenerator as OpenAIResponsesChatGenerator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
