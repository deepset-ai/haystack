# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "citation_consistency": ["Citation", "CitationConsistencyChecker"],
    "json_schema": ["JsonSchemaValidator"],
}

if TYPE_CHECKING:
    from .citation_consistency import Citation as Citation
    from .citation_consistency import CitationConsistencyChecker as CitationConsistencyChecker
    from .json_schema import JsonSchemaValidator as JsonSchemaValidator

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
