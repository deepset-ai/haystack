from haystack.lazy_imports import LazyImport
from haystack.nodes.file_converter.base import BaseConverter

from haystack.nodes.file_converter.csv import CsvTextConverter
from haystack.nodes.file_converter.docx import DocxToTextConverter
from haystack.nodes.file_converter.json import JsonConverter
from haystack.nodes.file_converter.tika import TikaConverter, TikaXHTMLParser
from haystack.nodes.file_converter.txt import TextConverter
from haystack.nodes.file_converter.azure import AzureConverter
from haystack.nodes.file_converter.parsr import ParsrConverter


try:
    with LazyImport() as fitz_import:
        # Try to use PyMuPDF, if not available fall back to xpdf
        from haystack.nodes.file_converter.pdf import PDFToTextConverter  # type: ignore

    fitz_import.check()
except (ModuleNotFoundError, ImportError):
    from haystack.nodes.file_converter.pdf_xpdf import PDFToTextConverter  # type: ignore  # pylint: disable=reimported,ungrouped-imports

from haystack.nodes.file_converter.markdown import MarkdownConverter
from haystack.nodes.file_converter.image import ImageToTextConverter
