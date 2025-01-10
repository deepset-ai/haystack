# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.converters.azure import AzureOCRDocumentConverter
    from haystack.components.converters.csv import CSVToDocument
    from haystack.components.converters.docx import DOCXMetadata, DOCXToDocument
    from haystack.components.converters.html import HTMLToDocument
    from haystack.components.converters.json import JSONConverter
    from haystack.components.converters.markdown import MarkdownToDocument
    from haystack.components.converters.openapi_functions import OpenAPIServiceToFunctions
    from haystack.components.converters.output_adapter import OutputAdapter
    from haystack.components.converters.pdfminer import PDFMinerToDocument
    from haystack.components.converters.pptx import PPTXToDocument
    from haystack.components.converters.pypdf import PyPDFToDocument
    from haystack.components.converters.tika import TikaDocumentConverter
    from haystack.components.converters.txt import TextFileToDocument


_lazy_imports = {
    "TextFileToDocument": "haystack.components.converters.txt",
    "TikaDocumentConverter": "haystack.components.converters.tika",
    "AzureOCRDocumentConverter": "haystack.components.converters.txt",
    "PyPDFToDocument": "haystack.components.converters.pypdf",
    "PDFMinerToDocument": "haystack.components.converters.pdfminer",
    "HTMLToDocument": "haystack.components.converters.html",
    "MarkdownToDocument": "haystack.components.converters.markdown",
    "OpenAPIServiceToFunctions": "haystack.components.converters.openapi_functions",
    "OutputAdapter": "haystack.components.converters.output_adapter",
    "DOCXToDocument": "haystack.components.converters.docx",
    "DOCXMetadata": "haystack.components.converters.docx",
    "PPTXToDocument": "haystack.components.converters.pptx",
    "CSVToDocument": "haystack.components.converters.csv",
    "JSONConverter": "haystack.components.converters.json",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
