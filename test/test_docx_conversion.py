import logging
from pathlib import Path

from haystack.indexing.file_converters.pdf import DocxToTextConverter

logger = logging.getLogger(__name__)


def test_extract_pages():
    converter = DocxToTextConverter()
    paragraphs = converter.extract_pages(file_path=Path("samples/sample_docx.docx"))
    assert len(text)==14 #Sampe has 14 Paragraphs
    assert text[1]=='' #Second Paragraph is Empty
