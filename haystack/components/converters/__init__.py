# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "azure": ["AzureOCRDocumentConverter"],
    "csv": ["CSVToDocument"],
    "docx": ["DOCXToDocument"],
    "html": ["HTMLToDocument"],
    "json": ["JSONConverter"],
    "markdown": ["MarkdownToDocument"],
    "msg": ["MSGToDocument"],
    "multi_file_converter": ["MultiFileConverter"],
    "openapi_functions": ["OpenAPIServiceToFunctions"],
    "output_adapter": ["OutputAdapter"],
    "pdfminer": ["PDFMinerToDocument"],
    "pptx": ["PPTXToDocument"],
    "pypdf": ["PyPDFToDocument"],
    "tika": ["TikaDocumentConverter"],
    "txt": ["TextFileToDocument"],
    "xlsx": ["XLSXToDocument"],
}

if TYPE_CHECKING:
    from .azure import AzureOCRDocumentConverter
    from .csv import CSVToDocument
    from .docx import DOCXToDocument
    from .html import HTMLToDocument
    from .json import JSONConverter
    from .markdown import MarkdownToDocument
    from .msg import MSGToDocument
    from .multi_file_converter import MultiFileConverter
    from .openapi_functions import OpenAPIServiceToFunctions
    from .output_adapter import OutputAdapter
    from .pdfminer import PDFMinerToDocument
    from .pptx import PPTXToDocument
    from .pypdf import PyPDFToDocument
    from .tika import TikaDocumentConverter
    from .txt import TextFileToDocument
    from .xlsx import XLSXToDocument

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
