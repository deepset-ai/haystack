from haystack.nodes.file_converter.base import BaseConverter
from haystack.nodes.file_converter.docx import DocxToTextConverter
from haystack.nodes.file_converter.image import ImageToTextConverter
from haystack.nodes.file_converter.markdown import MarkdownConverter
from haystack.nodes.file_converter.pdf import PDFToTextConverter, PDFToTextOCRConverter
from haystack.nodes.file_converter.tika import TikaConverter, TikaXHTMLParser
from haystack.nodes.file_converter.txt import TextConverter
from haystack.nodes.file_converter.azure import AzureConverter
