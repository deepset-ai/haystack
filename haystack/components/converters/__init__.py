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
    from .azure import AzureOCRDocumentConverter as AzureOCRDocumentConverter
    from .csv import CSVToDocument as CSVToDocument
    from .docx import DOCXToDocument as DOCXToDocument
    from .html import HTMLToDocument as HTMLToDocument
    from .json import JSONConverter as JSONConverter
    from .markdown import MarkdownToDocument as MarkdownToDocument
    from .msg import MSGToDocument as MSGToDocument
    from .multi_file_converter import MultiFileConverter as MultiFileConverter
    from .openapi_functions import OpenAPIServiceToFunctions as OpenAPIServiceToFunctions
    from .output_adapter import OutputAdapter as OutputAdapter
    from .pdfminer import PDFMinerToDocument as PDFMinerToDocument
    from .pptx import PPTXToDocument as PPTXToDocument
    from .pypdf import PyPDFToDocument as PyPDFToDocument
    from .tika import TikaDocumentConverter as TikaDocumentConverter
    from .txt import TextFileToDocument as TextFileToDocument
    from .xlsx import XLSXToDocument as XLSXToDocument

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
