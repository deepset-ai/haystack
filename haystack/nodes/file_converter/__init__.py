from haystack.nodes.file_converter.base import BaseConverter

from haystack.utils.import_utils import safe_import

from haystack.nodes.file_converter.docx import DocxToTextConverter
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
PDFToTextConverter = safe_import(
    "haystack.nodes.file_converter.pdf", "PDFToTextConverter", "ocr"
)  # Has optional dependencies
PDFToTextOCRConverter = safe_import(
    "haystack.nodes.file_converter.pdf", "PDFToTextOCRConverter", "ocr"
)  # Has optional dependencies
