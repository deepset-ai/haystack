from pathlib import Path

import pytest

from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.tika import TikaConverter


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_convert(Converter, xpdf_fixture):
    converter = Converter()
    document = converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    pages = document["text"].split("\f")
    assert len(pages) == 4  # the sample PDF file has four pages.
    assert pages[0] != ""  # the page 1 of PDF contains text.
    assert pages[2] == ""  # the page 3 of PDF file is empty.


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_table_removal(Converter, xpdf_fixture):
    converter = Converter(remove_numeric_tables=True)
    document = converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    pages = document["text"].split("\f")
    # assert numeric rows are removed from the table.
    assert "324" not in pages[0]
    assert "54x growth" not in pages[0]

    # assert text is retained from the document.
    assert "Adobe Systems made the PDF specification available free of charge in 1993." in pages[0].replace("\n", "")


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_language_validation(Converter, xpdf_fixture, caplog):
    converter = Converter(valid_languages=["en"])
    converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['en']." not in caplog.text

    converter = Converter(valid_languages=["de"])
    converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['de']." in caplog.text


