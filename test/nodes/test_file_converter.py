import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List
from unittest.mock import patch

import pandas as pd
import pytest

from haystack import Document
from haystack.nodes import (
    AzureConverter,
    CsvTextConverter,
    DocxToTextConverter,
    JsonConverter,
    MarkdownConverter,
    ParsrConverter,
    PDFToTextConverter,
    PreProcessor,
    TextConverter,
    TikaConverter,
)

from ..conftest import fail_at_version


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_convert(Converter, samples_path):
    converter = Converter()
    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_1.pdf")[0]["documents"][0]
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


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_command_whitespaces(Converter, samples_path):
    converter = Converter()

    document = converter.run(file_paths=samples_path / "pdf" / "sample pdf file with spaces on file name.pdf")[0][
        "documents"
    ][0]
    assert "ɪ" in document.content


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_encoding(Converter, samples_path):
    converter = Converter()

    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_5.pdf")[0]["documents"][0]
    assert "Ж" in document.content

    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_2.pdf")[0]["documents"][0]
    assert "ɪ" in document.content


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_sort_by_position(Converter, samples_path):
    converter = Converter(sort_by_position=True)

    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_3.pdf")[0]
    assert str(document.content).startswith("This is the second test sentence.")


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_ligatures(Converter, samples_path):
    converter = Converter()

    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_2.pdf")[0]["documents"][0]
    assert "ﬀ" not in document.content
    assert "ɪ" in document.content

    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_2.pdf", known_ligatures={})[0]["documents"][
        0
    ]
    assert "ﬀ" in document.content
    assert "ɪ" in document.content

    document = converter.run(file_paths=samples_path / "pdf" / "sample_pdf_2.pdf", known_ligatures={"ɪ": "i"})[0][
        "documents"
    ][0]
    assert "ﬀ" in document.content
    assert "ɪ" not in document.content


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_page_range(Converter, samples_path):
    converter = Converter()
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf", start_page=2)[0]
    pages = document.content.split("\f")

    assert (
        len(pages) == 4
    )  # the sample PDF file has four pages, we skipped first (but we wanna correct number of pages)
    assert pages[0] == ""  # the page 1 was skipped.
    assert pages[1] != ""  # the page 2 is not empty.
    assert pages[2] == ""  # the page 3 is empty.


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_page_range_numbers(Converter, samples_path):
    converter = Converter()
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf", start_page=2)[0]

    preprocessor = PreProcessor(
        split_by="word", split_length=5, split_overlap=0, split_respect_sentence_boundary=False, add_page_number=True
    )
    documents = preprocessor.process([document])

    assert documents[1].meta["page"] == 4


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_parallel(Converter, samples_path):
    converter = Converter(multiprocessing=True)
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_6.pdf")[0]

    pages = document.content.split("\f")

    assert pages[0] == "This is the page 1 of the document."
    assert pages[-1] == "This is the page 50 of the document."


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_parallel_page_range(Converter, samples_path):
    converter = Converter(multiprocessing=True)
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_6.pdf", start_page=2)[0]

    pages = document.content.split("\f")

    assert pages[0] == ""
    assert len(pages) == 50


@pytest.mark.unit
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_parallel_sort_by_position(Converter, samples_path):
    converter = Converter(multiprocessing=True, sort_by_position=True)
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_6.pdf")[0]

    pages = document.content.split("\f")

    assert pages[0] == "This is the page 1 of the document."
    assert pages[-1] == "This is the page 50 of the document."


@pytest.mark.integration
@pytest.mark.parametrize("Converter", [PDFToTextConverter])
def test_pdf_parallel_ocr(Converter, samples_path):
    converter = Converter(multiprocessing=True, sort_by_position=True, ocr="full", ocr_language="eng")
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_6.pdf")[0]

    pages = document.content.split("\f")

    assert pages[0] == "This is the page 1 of the document."
    assert pages[-1] == "This is the page 50 of the document."


@fail_at_version(1, 18)
def test_deprecated_encoding():
    with pytest.warns(DeprecationWarning):
        converter = PDFToTextConverter(encoding="utf-8")


@fail_at_version(1, 18)
def test_deprecated_encoding_in_convert_method(samples_path):
    converter = PDFToTextConverter()
    with pytest.warns(DeprecationWarning):
        converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf", encoding="utf-8")


@fail_at_version(1, 18)
def test_deprecated_keep_physical_layout():
    with pytest.warns(DeprecationWarning):
        converter = PDFToTextConverter(keep_physical_layout=True)


@fail_at_version(1, 18)
def test_deprecated_keep_physical_layout_in_convert_method(samples_path):
    converter = PDFToTextConverter()
    with pytest.warns(DeprecationWarning):
        converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf", keep_physical_layout=True)


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_table_removal(Converter, samples_path):
    converter = Converter(remove_numeric_tables=True)
    document = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf")[0]
    pages = document.content.split("\f")
    # assert numeric rows are removed from the table.
    assert "324" not in pages[0]
    assert "54x growth" not in pages[0]


@pytest.mark.tika
@pytest.mark.parametrize("Converter", [PDFToTextConverter, TikaConverter])
def test_language_validation(Converter, caplog, samples_path):
    converter = Converter(valid_languages=["en"])
    converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf")
    assert "sample_pdf_1.pdf is not one of ['en']." not in caplog.text

    converter = Converter(valid_languages=["de"])
    converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf")
    assert "sample_pdf_1.pdf is not one of ['de']." in caplog.text


@pytest.mark.unit
def test_docx_converter(samples_path):
    converter = DocxToTextConverter()
    document = converter.convert(file_path=samples_path / "docx" / "sample_docx.docx")[0]
    assert document.content.startswith("Sample Docx File")


@pytest.mark.unit
def test_markdown_converter(samples_path):
    converter = MarkdownConverter()
    document = converter.convert(file_path=samples_path / "markdown" / "sample.md")[0]
    assert document.content.startswith("\nWhat to build with Haystack")
    assert "# git clone https://github.com/deepset-ai/haystack.git" not in document.content


@pytest.mark.unit
def test_markdown_converter_headline_extraction(samples_path):
    expected_headlines = [
        ("What to build with Haystack", 1),
        ("Core Features", 1),
        ("Quick Demo", 1),
        ("2nd level headline for testing purposes", 2),
        ("3rd level headline for testing purposes", 3),
    ]

    converter = MarkdownConverter(extract_headlines=True, remove_code_snippets=False)
    document = converter.convert(file_path=samples_path / "markdown" / "sample.md")[0]

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


@pytest.mark.unit
def test_markdown_converter_frontmatter_to_meta(samples_path):
    converter = MarkdownConverter(add_frontmatter_to_meta=True)
    document = converter.convert(file_path=samples_path / "markdown" / "sample.md")[0]
    assert document.meta["type"] == "intro"
    assert document.meta["date"] == "1.1.2023"


@pytest.mark.unit
def test_markdown_converter_remove_code_snippets(samples_path):
    converter = MarkdownConverter(remove_code_snippets=False)
    document = converter.convert(file_path=samples_path / "markdown" / "sample.md")[0]
    assert document.content.startswith("pip install farm-haystack")


def test_azure_converter(samples_path):
    # Check if Form Recognizer endpoint and credential key in environment variables
    if "AZURE_FORMRECOGNIZER_ENDPOINT" in os.environ and "AZURE_FORMRECOGNIZER_KEY" in os.environ:
        converter = AzureConverter(
            endpoint=os.environ["AZURE_FORMRECOGNIZER_ENDPOINT"],
            credential_key=os.environ["AZURE_FORMRECOGNIZER_KEY"],
            save_json=True,
        )

        docs = converter.convert(file_path=samples_path / "pdf" / "sample_pdf_1.pdf")
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
def test_parsr_converter(samples_path):
    converter = ParsrConverter()

    docs = converter.convert(file_path=str((samples_path / "pdf" / "sample_pdf_1.pdf").absolute()))
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
def test_parsr_converter_headline_extraction(samples_path):
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

    docs = converter.convert(file_path=str((samples_path / "pdf" / "sample_pdf_4.pdf").absolute()))
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


@pytest.mark.integration
@pytest.mark.skipif(sys.platform in ["win32", "cygwin"], reason="Parsr not running on Windows CI")
def test_parsr_converter_list_mapping(samples_path):
    # This exact line(without line break characters) only exists in the list object we want to make sure it's being mapped correctly
    expected_list_line = "Maecenas tincidunt est efficitur ligula euismod, sit amet ornare est vulputate."

    converter = ParsrConverter()

    docs = converter.convert(file_path=str((samples_path / "pdf" / "sample_pdf_4.pdf").absolute()))
    assert len(docs) == 2
    assert docs[1].content_type == "text"
    assert expected_list_line in docs[1].content


@pytest.mark.unit
def test_id_hash_keys_from_pipeline_params(samples_path):
    doc_path = samples_path / "docs" / "doc_1.txt"
    meta_1 = {"key": "a"}
    meta_2 = {"key": "b"}
    meta = [meta_1, meta_2]

    converter = TextConverter()
    output, _ = converter.run(file_paths=[doc_path, doc_path], meta=meta, id_hash_keys=["content", "meta"])
    documents = output["documents"]
    unique_ids = set(d.id for d in documents)

    assert len(documents) == 2
    assert len(unique_ids) == 2


@pytest.mark.unit
def write_as_csv(data: List[List[str]], file_path: Path):
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_csv_to_document_with_one_column(tmp_path):
    node = CsvTextConverter()
    csv_path = tmp_path / "csv_qa_with_wrong_headers.csv"
    rows = [["question"], ["What is Haystack ?"]]
    write_as_csv(rows, csv_path)

    with pytest.raises(ValueError, match="The CSV must contain two columns named 'question' and 'answer'"):
        node.run(file_paths=csv_path)


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestJsonConverter:
    JSON_FILE_NAME = "json_normal.json"
    JSONL_FILE_NAME = "json_normal.jsonl"
    JSON_SINGLE_LINE_FILE_NAME = "json_all_single.json"
    JSONL_LIST_LINE_FILE_NAME = "json_list_line.jsonl"
    JSON_INVALID = "json_invalid.json"

    @classmethod
    @pytest.fixture(autouse=True)
    def setup_class(cls, tmp_path):
        # Setup the documents
        # Note: We are tying the behavior of `JsonConverter`
        # to that of the `to_dict()` method on the `Document`
        documents = [
            Document(
                content=pd.DataFrame(
                    [["C", "Yes", "No"], ["Haskell", "No", "No"], ["Python", "Yes", "Yes"]],
                    columns=["Language", "Imperative", "OO"],
                ),
                content_type="table",
                meta={"context": "Programming Languages", "page": 2},
            ),
            Document(
                content="Programming languages are used for controlling the behavior of a machine (often a computer).",
                content_type="text",
                meta={"context": "Programming Languages", "page": 1},
            ),
            Document(
                content=pd.DataFrame(
                    [["C", 1, 1], ["Python", 6, 6.5]], columns=["Language", "Statements ratio", "Line ratio"]
                ),
                content_type="table",
                meta={"context": "Expressiveness", "page": 3},
            ),
        ]

        doc_dicts_list = [d.to_dict() for d in documents]

        json_path = tmp_path / TestJsonConverter.JSON_FILE_NAME
        with open(json_path, "w") as f:
            json.dump(doc_dicts_list, f)

        jsonl_path = tmp_path / TestJsonConverter.JSONL_FILE_NAME
        with open(jsonl_path, "w") as f:
            for doc in doc_dicts_list:
                f.write(json.dumps(doc) + "\n")

        # json but everything written in a single line
        json_single_path = tmp_path / TestJsonConverter.JSON_SINGLE_LINE_FILE_NAME
        with open(json_single_path, "w") as f:
            f.write(json.dumps(doc_dicts_list))

        # Two lines (jsonl) but each line contains a list of dict instead of dict
        jsonl_list_line_path = tmp_path / TestJsonConverter.JSONL_LIST_LINE_FILE_NAME
        with open(jsonl_list_line_path, "w") as f:
            for doc in [doc_dicts_list[:2], doc_dicts_list[2:3]]:
                f.write(json.dumps(doc) + "\n")

        json_invalid_path = tmp_path / TestJsonConverter.JSON_INVALID
        with open(json_invalid_path, "w") as f:
            f.write("{an invalid json string}")

    def _assert_docs_okay(self, docs):
        # Two table docs and one text doc
        # [table, text, table]
        assert len(docs) == 3
        assert all(doc.meta["topic"] == "programming" for doc in docs)
        # "context" in metadata should have been overwritten to be "PL" instead of "Programming Languages"
        assert all(doc.meta["context"] == "PL" for doc in docs)
        assert all(d.content_type == expected for d, expected in zip(docs, ("table", "text", "table")))

        # Text doc test
        assert (
            docs[1].content
            == "Programming languages are used for controlling the behavior of a machine (often a computer)."
        )

        # Table doc tests
        assert isinstance(docs[0].content, pd.DataFrame)
        assert docs[0].content.shape == (3, 3)

        assert isinstance(docs[2].content, pd.DataFrame)
        assert docs[2].content.shape == (2, 3)

    def test_json_to_documents(self, tmp_path):
        json_path = tmp_path / TestJsonConverter.JSON_FILE_NAME

        converter = JsonConverter()
        docs = converter.convert(json_path, meta={"topic": "programming", "context": "PL"})

        self._assert_docs_okay(docs)

    def test_json_to_documents_single_line(self, tmp_path):
        json_path = tmp_path / TestJsonConverter.JSON_SINGLE_LINE_FILE_NAME

        converter = JsonConverter()
        docs = converter.convert(json_path, meta={"topic": "programming", "context": "PL"})

        self._assert_docs_okay(docs)

    def test_jsonl_to_documents(self, tmp_path):
        jsonl_path = tmp_path / TestJsonConverter.JSONL_FILE_NAME

        converter = JsonConverter()
        docs = converter.convert(jsonl_path, meta={"topic": "programming", "context": "PL"})

        self._assert_docs_okay(docs)

    def test_jsonl_to_documents_list_line(self, tmp_path):
        jsonl_path = tmp_path / TestJsonConverter.JSONL_LIST_LINE_FILE_NAME

        converter = JsonConverter()
        docs = converter.convert(jsonl_path, meta={"topic": "programming", "context": "PL"})

        self._assert_docs_okay(docs)

    def test_json_invalid(self, tmp_path):
        json_path = tmp_path / TestJsonConverter.JSON_INVALID

        converter = JsonConverter()
        with pytest.raises(json.JSONDecodeError) as excinfo:
            converter.convert(json_path)

        # Assert filename is in the error message
        assert TestJsonConverter.JSON_INVALID in str(excinfo.value)
