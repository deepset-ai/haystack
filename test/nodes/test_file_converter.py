from typing import List

import os
import sys
from pathlib import Path
import subprocess
import csv

import pytest

from haystack import Document
from haystack.nodes import (
    MarkdownConverter,
    DocxToTextConverter,
    PDFToTextConverter,
    PDFToTextOCRConverter,
    TikaConverter,
    AzureConverter,
    ParsrConverter,
    TextConverter,
    CsvTextConverter,
)

from ..conftest import SAMPLES_PATH


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter, PDFToTextOCRConverter])
def test_convert(Converter):
    converter = Converter()
    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")[0]["documents"][0]
    pages = document.content.split("\f")

    assert (
        len(pages) != 1 and pages[0] != ""
    ), f'{type(converter).__name__} did return a single empty page indicating a potential issue with your installed poppler version. Try installing via "conda install -c conda-forge poppler" and check test_pdftoppm_command_format()'

    assert len(pages) == 4  # the sample PDF file has four pages.
    assert pages[0] != ""  # the page 1 of PDF contains text.
    assert pages[2] == ""  # the page 3 of PDF file is empty.
    # assert text is retained from the document.
    # As whitespace can differ (\n," ", etc.), we standardize all to simple whitespace
    page_standard_whitespace = " ".join(pages[0].split())
    assert "Adobe Systems made the PDF specification available free of charge in 1993." in page_standard_whitespace


# Marked as integration because it uses poppler, which is not installed in the unit tests suite
@pytest.mark.integration
@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Poppler not installed on Windows CI")
def test_pdftoppm_command_format():
    # Haystack's PDFToTextOCRConverter uses pdf2image, which calls pdftoppm internally.
    # Some installations of pdftoppm are incompatible with Haystack and won't raise an error but just return empty converted documents
    # This test runs pdftoppm directly to check whether pdftoppm accepts the command format that pdf2image uses in Haystack
    proc = subprocess.Popen(
        ["pdftoppm", f"{SAMPLES_PATH}/pdf/sample_pdf_1.pdf"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    # If usage info of pdftoppm is sent to stderr then it's because Haystack's pdf2image uses an incompatible command format
    assert (
        not err
    ), 'Your installation of poppler is incompatible with Haystack. Try installing via "conda install -c conda-forge poppler"'


@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_command_whitespaces(Converter):
    converter = Converter()

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample pdf file with spaces on file name.pdf")[0][
        "documents"
    ][0]
    assert "ɪ" in document.content


@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_encoding(Converter):
    converter = Converter()

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf")[0]["documents"][0]
    assert "ɪ" in document.content

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf", encoding="Latin1")[0]["documents"][0]
    assert "ɪ" not in document.content


@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_layout(Converter):
    converter = Converter(keep_physical_layout=True)

    document = converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_3.pdf")[0]
    assert str(document.content).startswith("This is the second test sentence.")


@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_ligatures(Converter):
    converter = Converter()

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf")[0]["documents"][0]
    assert "ﬀ" not in document.content
    assert "ɪ" in document.content

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf", known_ligatures={})[0]["documents"][
        0
    ]
    assert "ﬀ" in document.content
    assert "ɪ" in document.content

    document = converter.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_2.pdf", known_ligatures={"ɪ": "i"})[0][
        "documents"
    ][0]
    assert "ﬀ" in document.content
    assert "ɪ" not in document.content


@pytest.mark.parametrize("Converter", [PDFToTextConverter, PDFToTextOCRConverter])
def test_page_range(Converter):
    converter = Converter()
    document = converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf", start_page=2)[0]
    pages = document.content.split("\f")

    assert (
        len(pages) == 4
    )  # the sample PDF file has four pages, we skipped first (but we wanna correct number of pages)
    assert pages[0] == ""  # the page 1 was skipped.
    assert pages[1] != ""  # the page 2 is not empty.
    assert pages[2] == ""  # the page 3 is empty.


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_table_removal(Converter):
    converter = Converter(remove_numeric_tables=True)
    document = converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")[0]
    pages = document.content.split("\f")
    # assert numeric rows are removed from the table.
    assert "324" not in pages[0]
    assert "54x growth" not in pages[0]


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_language_validation(Converter, caplog):
    converter = Converter(valid_languages=["en"])
    converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    assert "sample_pdf_1.pdf is not one of ['en']." not in caplog.text

    converter = Converter(valid_languages=["de"])
    converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    assert "sample_pdf_1.pdf is not one of ['de']." in caplog.text


def test_docx_converter():
    converter = DocxToTextConverter()
    document = converter.convert(file_path=SAMPLES_PATH / "docx" / "sample_docx.docx")[0]
    assert document.content.startswith("Sample Docx File")


def test_markdown_converter():
    converter = MarkdownConverter()
    document = converter.convert(file_path=SAMPLES_PATH / "markdown" / "sample.md")[0]
    assert document.content.startswith("What to build with Haystack")


def test_markdown_converter_headline_extraction():
    expected_headlines = [
        ("What to build with Haystack", 1),
        ("Core Features", 1),
        ("Quick Demo", 1),
        ("2nd level headline for testing purposes", 2),
        ("3rd level headline for testing purposes", 3),
    ]

    converter = MarkdownConverter(extract_headlines=True, remove_code_snippets=False)
    document = converter.convert(file_path=SAMPLES_PATH / "markdown" / "sample.md")[0]

    # Check if correct number of headlines are extracted
    assert len(document.meta["headlines"]) == 5
    for extracted_headline, (expected_headline, expected_level) in zip(document.meta["headlines"], expected_headlines):
        # Check if correct headline and level is extracted
        assert extracted_headline["headline"] == expected_headline
        assert extracted_headline["level"] == expected_level

        # Check if correct start_idx is extracted
        start_idx = extracted_headline["start_idx"]
        hl_len = len(extracted_headline["headline"])
        assert extracted_headline["headline"] == document.content[start_idx : start_idx + hl_len]


def test_markdown_converter_frontmatter_to_meta():
    converter = MarkdownConverter(add_frontmatter_to_meta=True)
    document = converter.convert(file_path=SAMPLES_PATH / "markdown" / "sample.md")[0]
    assert document.meta["type"] == "intro"
    assert document.meta["date"] == "1.1.2023"


def test_azure_converter():
    # Check if Form Recognizer endpoint and credential key in environment variables
    if "AZURE_FORMRECOGNIZER_ENDPOINT" in os.environ and "AZURE_FORMRECOGNIZER_KEY" in os.environ:
        converter = AzureConverter(
            endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"],
            credential_key=os.environ["AZURE_FORMRECOGNIZER_KEY"],
            save_json=True,
        )

        docs = converter.convert(file_path=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
        assert len(docs) == 2
        assert docs[0].content_type == "table"
        assert docs[0].content.shape[0] == 4  # number of rows
        assert docs[0].content.shape[1] == 5  # number of columns, Form Recognizer assumes there are 5 columns
        assert list(docs[0].content.columns) == ["", "Column 1", "", "Column 2", "Column 3"]
        assert list(docs[0].content.iloc[3]) == ["D", "$54.35", "", "$6345.", ""]
        assert (
            docs[0].meta["preceding_context"] == "specification. These proprietary technologies are not "
            "standardized and their\nspecification is published only on "
            "Adobe's website. Many of them are also not\nsupported by "
            "popular third-party implementations of PDF."
        )
        assert docs[0].meta["following_context"] == ""
        assert docs[0].meta["page"] == 1

        assert docs[1].content_type == "text"
        assert docs[1].content.startswith("A sample PDF file")


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Parsr not running on Windows CI")
def test_parsr_converter():
    converter = ParsrConverter()

    docs = converter.convert(file_path=str((SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf").absolute()))
    assert len(docs) == 2
    assert docs[0].content_type == "table"
    assert docs[0].content.shape[0] == 4  # number of rows
    assert docs[0].content.shape[1] == 4
    assert list(docs[0].content.columns) == ["", "Column 1", "Column 2", "Column 3"]
    assert list(docs[0].content.iloc[3]) == ["D", "$54.35", "$6345.", ""]
    assert (
        docs[0].meta["preceding_context"] == "speciﬁcation. These proprietary technologies are not "
        "standardized and their\nspeciﬁcation is published only on "
        "Adobe's website. Many of them are also not\nsupported by popular "
        "third-party implementations of PDF."
    )
    assert docs[0].meta["following_context"] == ""
    assert docs[0].meta["page"] == 1

    assert docs[1].content_type == "text"
    assert docs[1].content.startswith("A sample PDF ﬁle")
    assert docs[1].content.endswith("Page 4 of Sample PDF\n… the page 3 is empty.")


@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Parsr not running on Windows CI")
def test_parsr_converter_headline_extraction():
    expected_headlines = [
        [("Lorem ipsum", 1), ("Cras fringilla ipsum magna, in fringilla dui commodo\na.", 2)],
        [
            ("Lorem ipsum", 1),
            ("Lorem ipsum dolor sit amet, consectetur adipiscing\nelit. Nunc ac faucibus odio.", 2),
            ("Cras fringilla ipsum magna, in fringilla dui commodo\na.", 2),
            ("Lorem ipsum dolor sit amet, consectetur adipiscing\nelit.", 2),
            ("Maecenas mauris lectus, lobortis et purus mattis, blandit\ndictum tellus.", 2),
            ("In eleifend velit vitae libero sollicitudin euismod.", 2),
        ],
    ]

    converter = ParsrConverter()

    docs = converter.convert(file_path=str((SAMPLES_PATH / "pdf" / "sample_pdf_4.pdf").absolute()))
    assert len(docs) == 2

    for doc, expectation in zip(docs, expected_headlines):
        for extracted_headline, (expected_headline, expected_level) in zip(doc.meta["headlines"], expectation):
            # Check if correct headline and level is extracted
            assert extracted_headline["headline"] == expected_headline
            assert extracted_headline["level"] == expected_level

            # Check if correct start_idx is extracted
            if doc.content_type == "text":
                start_idx = extracted_headline["start_idx"]
                hl_len = len(extracted_headline["headline"])
                assert extracted_headline["headline"] == doc.content[start_idx : start_idx + hl_len]


def test_id_hash_keys_from_pipeline_params():
    doc_path = SAMPLES_PATH / "docs" / "doc_1.txt"
    meta_1 = {"key": "a"}
    meta_2 = {"key": "b"}
    meta = [meta_1, meta_2]

    converter = TextConverter()
    output, _ = converter.run(file_paths=[doc_path, doc_path], meta=meta, id_hash_keys=["content", "meta"])
    documents = output["documents"]
    unique_ids = set(d.id for d in documents)

    assert len(documents) == 2
    assert len(unique_ids) == 2


def write_as_csv(data: List[List[str]], file_path: Path):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


@pytest.mark.integration
def test_csv_to_document_with_qa_headers(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_headers.csv"
    rows = [
        ["question", "answer"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications."],
    ]
    write_as_csv(rows, csv_path)

    output, edge = node.run(file_paths=csv_path)
    assert edge == "output_1"
    assert "documents" in output
    assert len(output["documents"]) == 1

    doc = output["documents"][0]
    assert isinstance(doc, Document)
    assert doc.content == "What is Haystack ?"
    assert doc.meta["answer"] == "Haystack is an NLP Framework to use transformers in your Applications."


@pytest.mark.integration
def test_csv_to_document_with_wrong_qa_headers(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [
        ["wrong", "headers"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications."],
    ]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.integration
def test_csv_to_document_with_one_wrong_qa_headers(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [
        ["wrong", "answers"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications."],
    ]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.integration
def test_csv_to_document_with_another_wrong_qa_headers(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [
        ["question", "wrong"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications."],
    ]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.integration
def test_csv_to_document_with_one_column(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [["question"], ["What is Haystack ?"]]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.integration
def test_csv_to_document_with_three_columns(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [
        ["question", "answer", "notes"],
        ["What is Haystack ?", "Haystack is an NLP Framework to use transformers in your Applications.", "verified"],
    ]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.integration
def test_csv_to_document_many_files(tmp_path):
    csv_paths = []
    for i in range(5):
        node = CsvTextConverter()
        csv_path = tmp_path / f"{i}_csv_qa_with_headers.csv"
        csv_paths.append(csv_path)
        rows = [
            ["question", "answer"],
            [
                f"{i}. What is Haystack ?",
                f"{i}. Haystack is an NLP Framework to use transformers in your Applications.",
            ],
        ]
        write_as_csv(rows, csv_path)

    output, edge = node.run(file_paths=csv_paths)
    assert edge == "output_1"
    assert "documents" in output
    assert len(output["documents"]) == 5

    for i in range(5):
        doc = output["documents"][i]
        assert isinstance(doc, Document)
        assert doc.content == f"{i}. What is Haystack ?"
        assert doc.meta["answer"] == f"{i}. Haystack is an NLP Framework to use transformers in your Applications."
