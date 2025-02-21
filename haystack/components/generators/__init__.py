# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "openai": ["OpenAIGenerator"],
    "azure": ["AzureOpenAIGenerator"],
    "hugging_face_local": ["HuggingFaceLocalGenerator"],
    "hugging_face_api": ["HuggingFaceAPIGenerator"],
    "openai_dalle": ["DALLEImageGenerator"],
}

if TYPE_CHECKING:
    from .azure import AzureOpenAIGenerator
    from .hugging_face_api import HuggingFaceAPIGenerator
    from .hugging_face_local import HuggingFaceLocalGenerator
    from .openai import OpenAIGenerator
    from .openai_dalle import DALLEImageGenerator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
