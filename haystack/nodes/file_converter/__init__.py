from haystack.nodes.file_converter.base import BaseConverter

from haystack.utils.import_utils import safe_import

from haystack.nodes.file_converter.csv import CsvTextConverter
from haystack.nodes.file_converter.docx import DocxToTextConverter
from haystack.nodes.file_converter.json import JsonConverter
from haystack.nodes.file_converter.tika import TikaConverter, TikaXHTMLParser
from haystack.nodes.file_converter.txt import TextConverter
from haystack.nodes.file_converter.azure import AzureConverter
from haystack.nodes.file_converter.parsr import ParsrConverter


MarkdownConverter = safe_import(
    "haystack.nodes.file_converter.markdown", "MarkdownConverter", "preprocessing"
)  # Has optional dependencies
ImageToTextConverter = safe_import(
    "haystack.nodes.file_converter.image", "ImageToTextConverter", "ocr"
)  # Has optional dependencies

# Try to use PyMuPDF, if not available fall back to xpdf
try:
    from haystack.nodes.file_converter.pdf import PDFToTextConverter
except ImportError:
    from haystack.nodes.file_converter.pdf_xpdf import PDFToTextConverter  # type: ignore
