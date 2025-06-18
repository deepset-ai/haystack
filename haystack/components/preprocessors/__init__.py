# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "csv_document_cleaner": ["CSVDocumentCleaner"],
    "csv_document_splitter": ["CSVDocumentSplitter"],
    "document_cleaner": ["DocumentCleaner"],
    "document_preprocessor": ["DocumentPreprocessor"],
    "document_splitter": ["DocumentSplitter"],
    "hierarchical_document_splitter": ["HierarchicalDocumentSplitter"],
    "recursive_splitter": ["RecursiveDocumentSplitter"],
    "text_cleaner": ["TextCleaner"],
}

if TYPE_CHECKING:
    from .csv_document_cleaner import CSVDocumentCleaner as CSVDocumentCleaner
    from .csv_document_splitter import CSVDocumentSplitter as CSVDocumentSplitter
    from .document_cleaner import DocumentCleaner as DocumentCleaner
    from .document_preprocessor import DocumentPreprocessor as DocumentPreprocessor
    from .document_splitter import DocumentSplitter as DocumentSplitter
    from .hierarchical_document_splitter import HierarchicalDocumentSplitter as HierarchicalDocumentSplitter
    from .recursive_splitter import RecursiveDocumentSplitter as RecursiveDocumentSplitter
    from .text_cleaner import TextCleaner as TextCleaner

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
