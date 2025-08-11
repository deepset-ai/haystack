# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "conditional_router": ["ConditionalRouter"],
    "document_length_router": ["DocumentLengthRouter"],
    "document_type_router": ["DocumentTypeRouter"],
    "file_type_router": ["FileTypeRouter"],
    "llm_messages_router": ["LLMMessagesRouter"],
    "metadata_router": ["MetadataRouter"],
    "text_language_router": ["TextLanguageRouter"],
    "transformers_text_router": ["TransformersTextRouter"],
    "zero_shot_text_router": ["TransformersZeroShotTextRouter"],
}

if TYPE_CHECKING:
    from .conditional_router import ConditionalRouter as ConditionalRouter
    from .document_length_router import DocumentLengthRouter as DocumentLengthRouter
    from .document_type_router import DocumentTypeRouter as DocumentTypeRouter
    from .file_type_router import FileTypeRouter as FileTypeRouter
    from .llm_messages_router import LLMMessagesRouter as LLMMessagesRouter
    from .metadata_router import MetadataRouter as MetadataRouter
    from .text_language_router import TextLanguageRouter as TextLanguageRouter
    from .transformers_text_router import TransformersTextRouter as TransformersTextRouter
    from .zero_shot_text_router import TransformersZeroShotTextRouter as TransformersZeroShotTextRouter
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
