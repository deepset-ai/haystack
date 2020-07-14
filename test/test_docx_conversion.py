import logging
from pathlib import Path

from haystack.indexing.file_converters.pdf import DocxToTextConverter

logger = logging.getLogger(__name__)

def test_extract_pages(xpdf_fixture):
    converter = DocxToTextConverter()
    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf")
    
