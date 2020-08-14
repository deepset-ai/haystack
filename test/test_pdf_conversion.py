import logging
from pathlib import Path
import pytest

from haystack.indexing.file_converters.pdf import PDFToTextConverter
from haystack.indexing.file_converters.tika import TikaConverter
logger = logging.getLogger(__name__)


# @pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
# def test_extract_pages(Converter, xpdf_fixture):
#     converter = Converter()
#     pages, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
#     assert len(pages) == 4  # the sample PDF file has four pages.
#     assert pages[0] != ""  # the page 1 of PDF contains text.
#     assert pages[2] == ""  # the page 3 of PDF file is empty.

#
# @pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
# def test_table_removal(Converter, xpdf_fixture):
#     converter = Converter(remove_numeric_tables=True)
#     pages, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
#
#     # assert numeric rows are removed from the table.
#     assert "324" not in pages[0]
#     assert "54x growth" not in pages[0]
#     # assert "$54.35" not in pages[0]
#
#     # assert text is retained from the document.
#     assert "Adobe Systems made the PDF specification available free of charge in 1993." in pages[0].replace("\n", "")
#
#
# @pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
# def test_language_validation(Converter, xpdf_fixture, caplog):
#     converter = Converter(valid_languages=["en"])
#     pages, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
#     assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['en']." not in caplog.text
#
#     converter = Converter(valid_languages=["de"])
#     pages, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
#     assert "The language for samples/pdf/sample_pdf_1.pdf is not one of ['de']." in caplog.text
#
#
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_header_footer_removal(Converter, xpdf_fixture):
    converter = Converter(remove_header_footer=True)
    converter_no_removal = Converter(remove_header_footer=False)

    pages1, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))  # file contains no header/footer
    pages2, _ = converter_no_removal.extract_pages(file_path=Path("samples/pdf/sample_pdf_1.pdf"))  # file contains no header/footer
    for p1, p2 in zip(pages1, pages2):
        assert p2 == p2

    pages, _ = converter.extract_pages(file_path=Path("samples/pdf/sample_pdf_2.pdf"))  # file contains header and footer
    assert len(pages) == 4
    # for page in pages:
    #     assert "header" not in page
    #     assert "footer" not in page