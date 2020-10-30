from pathlib import Path

import pytest

from haystack.file_converter.docx import DocxToTextConverter


@pytest.mark.tika
def test_convert():
    converter = DocxToTextConverter()
    document = converter.convert(file_path=Path("samples/docx/sample_docx.docx"))
    assert document["text"].startswith("Sample Docx File")
