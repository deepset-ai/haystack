from pathlib import Path

from haystack.file_converter.docx import DocxToTextConverter


def test_convert():
    converter = DocxToTextConverter()
    document = converter.convert(file_path=Path("samples/docx/sample_docx.docx"))
    assert document["text"].startswith("Sample Docx File")
