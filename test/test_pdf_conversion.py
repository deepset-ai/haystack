import logging
from pathlib import Path

from haystack.indexing.file_converters.pdftotext import PDFToTextConverter

logger = logging.getLogger(__name__)


def test_extract_pages(xpdf_fixture):
    converter = PDFToTextConverter()
    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert len(pages) == 4  # the sample PDF file has four pages.
    assert pages[0] != ""  # the page 1 of PDF contains text.
    assert pages[2] == ""  # the page 3 of PDF file is empty.


def test_table_removal(xpdf_fixture):
    converter = PDFToTextConverter(remove_numeric_tables=True)
    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))

    # assert numeric rows are removed from the table.
    assert "324" not in pages[0]
    assert "54x growth" not in pages[0]
    assert "$54.35" not in pages[0]

    # assert text is retained from the document.
    assert "Adobe Systems made the PDF specification available free of charge in 1993." in pages[0]


def test_language_validation(xpdf_fixture, caplog):
    converter = PDFToTextConverter(valid_languages=["en"])
    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['en']." not in caplog.text

    converter = PDFToTextConverter(valid_languages=["de"])
    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['de']." in caplog.text


def test_header_footer_removal(xpdf_fixture):
    converter = PDFToTextConverter(remove_header_footer=True)

    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))  # file contains no header/footer
    assert converter.find_header_footer(pages) is None

    pages = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_2.pdf"))  # file contains no header and footer
    assert converter.find_header_footer(pages) is not None
    for page in pages:
        assert "header" not in page
        assert "footer" not in page
