# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "llm_metadata_extractor": ["LLMMetadataExtractor"],
    "named_entity_extractor": ["NamedEntityAnnotation", "NamedEntityExtractor", "NamedEntityExtractorBackend"],
}

if TYPE_CHECKING:
    from .llm_metadata_extractor import LLMMetadataExtractor as LLMMetadataExtractor
    from .named_entity_extractor import NamedEntityAnnotation as NamedEntityAnnotation
    from .named_entity_extractor import NamedEntityExtractor as NamedEntityExtractor
    from .named_entity_extractor import NamedEntityExtractorBackend as NamedEntityExtractorBackend

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
