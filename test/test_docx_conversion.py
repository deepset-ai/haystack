from pathlib import Path

from haystack.indexing.file_converters.docx import DocxToTextConverter


def test_extract_pages():
    converter = DocxToTextConverter()
    paragraphs, _ = converter.extract_pages(file_path=Path("samples/docx/sample_docx.docx"))
    assert len(paragraphs) == 8  # Sample has 8 Paragraphs
    assert paragraphs[1] == 'The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.'
