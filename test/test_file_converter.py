from pathlib import Path
import os

import pytest

from haystack.nodes import MarkdownConverter, DocxToTextConverter, PDFToTextConverter, PDFToTextOCRConverter, \
    TikaConverter, AzureConverter


@pytest.mark.tika
@pytest.mark.parametrize(
    # "Converter", [PDFToTextConverter, TikaConverter, PDFToTextOCRConverter]
    "Converter", [PDFToTextOCRConverter]
)
def test_convert(Converter):
    converter = Converter()
    document = converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))[0]
    pages = document["content"].split("\f")
    assert len(pages) == 4  # the sample PDF file has four pages.
    assert pages[0] != ""  # the page 1 of PDF contains text.
    assert pages[2] == ""  # the page 3 of PDF file is empty.
    # assert text is retained from the document.
    # As whitespace can differ (\n," ", etc.), we standardize all to simple whitespace
    page_standard_whitespace = " ".join(pages[0].split())
    assert (
        "Adobe Systems made the PDF specification available free of charge in 1993."
        in page_standard_whitespace
    )


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_table_removal(Converter):
    converter = Converter(remove_numeric_tables=True)
    document = converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))[0]
    pages = document["content"].split("\f")
    # assert numeric rows are removed from the table.
    assert "324" not in pages[0]
    assert "54x growth" not in pages[0]


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_language_validation(Converter, caplog):
    converter = Converter(valid_languages=["en"])
    converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert (
        "The language for samples/pdf/sample_pdf_1.pdf is not one of ['en']."
        not in caplog.text
    )

    converter = Converter(valid_languages=["de"])
    converter.convert(file_path=Path("samples/pdf/sample_pdf_1.pdf"))
    assert (
        "The language for samples/pdf/sample_pdf_1.pdf is not one of ['de']."
        in caplog.text
    )


def test_docx_converter():
    converter = DocxToTextConverter()
    document = converter.convert(file_path=Path("samples/docx/sample_docx.docx"))[0]
    assert document["content"].startswith("Sample Docx File")


def test_markdown_converter():
    converter = MarkdownConverter()
    document = converter.convert(file_path=Path("samples/markdown/sample.md"))[0]
    assert document["content"].startswith("What to build with Haystack")


def test_azure_converter():
    # Check if Form Recognizer endpoint and credential key in environment variables
    if "AZURE_FORMRECOGNIZER_ENDPOINT" in os.environ and "AZURE_FORMRECOGNIZER_KEY" in os.environ:
        converter = AzureConverter(endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"],
                                   credential_key=os.environ["AZURE_FORMRECOGNIZER_KEY"],
                                   save_json=True,
                                   )

        docs = converter.convert(file_path="samples/pdf/sample_pdf_1.pdf")
        assert len(docs) == 2
        assert docs[0]["content_type"] == "table"
        assert len(docs[0]["content"]) == 5  # number of rows
        assert len(docs[0]["content"][0]) == 5  # number of columns, Form Recognizer assumes there are 5 columns
        assert docs[0]["content"][0] == ['', 'Column 1', '', 'Column 2', 'Column 3']
        assert docs[0]["content"][4] == ['D', '$54.35', '', '$6345.', '']

        assert docs[1]["content_type"] == "text"
        assert docs[1]["content"].startswith("A sample PDF file")
