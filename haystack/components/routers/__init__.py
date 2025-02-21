# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "conditional_router": ["ConditionalRouter"],
    "file_type_router": ["FileTypeRouter"],
    "metadata_router": ["MetadataRouter"],
    "text_language_router": ["TextLanguageRouter"],
    "transformers_text_router": ["TransformersTextRouter"],
    "zero_shot_text_router": ["TransformersZeroShotTextRouter"],
}

if TYPE_CHECKING:
    from .conditional_router import ConditionalRouter
    from .file_type_router import FileTypeRouter
    from .metadata_router import MetadataRouter
    from .text_language_router import TextLanguageRouter
    from .transformers_text_router import TransformersTextRouter
    from .zero_shot_text_router import TransformersZeroShotTextRouter
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
