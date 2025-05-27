# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ByteStream
from haystack.components.converters.pdfminer import PDFMinerToDocument


class TestPDFMinerToDocument:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content

    def test_init_params_custom(self, test_files_path):
        """
        Test if init arguments are passed successfully to PDFMinerToDocument layout parameters
        """
        converter = PDFMinerToDocument(char_margin=0.5, all_texts=True, store_full_path=False)
        assert converter.layout_params.char_margin == 0.5
        assert converter.layout_params.all_texts is True
        assert converter.store_full_path is False

    def test_run_with_store_full_path_false(self, test_files_path):
        """
        Test if the component runs correctly with store_full_path=False
        """
        converter = PDFMinerToDocument(store_full_path=False)
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content
            assert doc.meta["file_path"] == "sample_pdf_1.pdf"

    def test_run_wrong_file_type(self, test_files_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = PDFMinerToDocument()

        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=sources)
            assert "Is this really a PDF?" in caplog.text

        docs = output["documents"]
        assert not docs

    def test_arg_is_none(self, test_files_path):
        """
        Test if the component runs correctly when an argument is None.
        """
        converter = PDFMinerToDocument(char_margin=None)
        assert converter.layout_params.char_margin is None

    def test_run_doc_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is supplied by the user.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        metadata = [{"file_name": "sample_pdf_2.pdf"}]
        results = converter.run(sources=sources, meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta["file_name"] == "sample_pdf_2.pdf"

    def test_incorrect_meta(self, test_files_path):
        """
        Test if the component raises an error when incorrect metadata is supplied by the user.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_3.pdf"]
        metadata = [{"file_name": "sample_pdf_3.pdf"}, {"file_name": "sample_pdf_2.pdf"}]
        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
            converter.run(sources=sources, meta=metadata)

    def test_run_bytestream_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object.
        """
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/pdf", "url": "test_url"})

        results = converter.run(sources=[stream])
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {"content_type": "text/pdf", "url": "test_url"}

    def test_run_bytestream_doc_overlapping_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is an overlap between the metadata received.

        The component should use the supplied metadata to overwrite the values if there is an overlap between the keys.
        """
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            byte_stream = file.read()
            # ByteStream has "url" present in metadata
            stream = ByteStream(byte_stream, meta={"content_type": "text/pdf", "url": "test_url_correct"})

        # "url" supplied by the user overwrites value present in metadata
        metadata = [{"file_name": "sample_pdf_2.pdf", "url": "test_url_new"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {"file_name": "sample_pdf_2.pdf", "content_type": "text/pdf", "url": "test_url_new"}

    def test_run_error_handling(self, caplog):
        """
        Test if the component correctly handles errors.
        """
        sources = ["non_existing_file.pdf"]
        converter = PDFMinerToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.pdf" in caplog.text
            assert results["documents"] == []

    def test_run_empty_document(self, caplog, test_files_path):
        sources = [test_files_path / "pdf" / "non_text_searchable.pdf"]
        converter = PDFMinerToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "PDFMinerToDocument could not extract text from the file" in caplog.text
            assert results["documents"][0].content == ""

            # Check that not only content is used when the returned document is initialized and doc id is generated
            assert results["documents"][0].meta["file_path"] == "non_text_searchable.pdf"
            assert results["documents"][0].id != Document(content="").id

    def test_run_detect_pages_and_split_by_passage(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        pdf_doc = converter.run(sources=sources)
        splitter = DocumentSplitter(split_length=1, split_by="page")
        docs = splitter.run(pdf_doc["documents"])
        assert len(docs["documents"]) == 4

    def test_run_detect_paragraphs_to_be_used_in_split_passage(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        pdf_doc = converter.run(sources=sources)
        splitter = DocumentSplitter(split_length=1, split_by="passage")
        docs = splitter.run(pdf_doc["documents"])

        assert len(docs["documents"]) == 29

        expected = (
            "\nA wiki (/ˈwɪki/ (About this soundlisten) WIK-ee) is a hypertext publication collaboratively"
            " \nedited and managed by its own audience directly using a web browser. A typical wiki \ncontains "
            "multiple pages for the subjects or scope of the project and may be either open \nto the public or "
            "limited to use within an organization for maintaining its internal knowledge \nbase. Wikis are "
            "enabled by wiki software, otherwise known as wiki engines. A wiki engine, \nbeing a form of a "
            "content management system, diﬀers from other web-based systems \nsuch as blog software, in that "
            "the content is created without any deﬁned owner or leader, \nand wikis have little inherent "
            "structure, allowing structure to emerge according to the \nneeds of the users.[1] \n\n"
        )
        assert docs["documents"][6].content == expected

    def test_detect_undecoded_cid_characters(self):
        """
        Test if the component correctly detects and reports undecoded CID characters in text.
        """
        converter = PDFMinerToDocument()

        # Test text with no CID characters
        text = "This is a normal text without any CID characters."
        result = converter.detect_undecoded_cid_characters(text)
        assert result["total_chars"] == len(text)
        assert result["cid_chars"] == 0
        assert result["percentage"] == 0

        # Test text with CID characters
        text = "Some text with (cid:123) and (cid:456) characters"
        result = converter.detect_undecoded_cid_characters(text)
        assert result["total_chars"] == len(text)
        assert result["cid_chars"] == len("(cid:123)") + len("(cid:456)")  # 18 characters total
        assert result["percentage"] == round((18 / len(text)) * 100, 2)

        # Test text with multiple consecutive CID characters
        text = "(cid:123)(cid:456)(cid:789)"
        result = converter.detect_undecoded_cid_characters(text)
        assert result["total_chars"] == len(text)
        assert result["cid_chars"] == len("(cid:123)(cid:456)(cid:789)")
        assert result["percentage"] == 100.0

        # Test empty text
        text = ""
        result = converter.detect_undecoded_cid_characters(text)
        assert result["total_chars"] == 0
        assert result["cid_chars"] == 0
        assert result["percentage"] == 0

    def test_pdfminer_logs_warning_for_cid_characters(self, caplog, monkeypatch):
        """
        Test if the component correctly logs a warning when undecoded CID characters are detected.
        """
        test_data = ByteStream(data=b"fake", meta={"file_path": "test.pdf"})

        def mock_converter(*args, **kwargs):
            return "This is text with (cid:123) and (cid:456) characters"

        def mock_extract_pages(*args, **kwargs):
            return ["mocked page"]

        with patch("haystack.components.converters.pdfminer.extract_pages", side_effect=mock_extract_pages):
            with patch.object(PDFMinerToDocument, "_converter", side_effect=mock_converter):
                with caplog.at_level(logging.WARNING):
                    converter = PDFMinerToDocument()
                    converter.run(sources=[test_data])
                    assert "Detected 18 undecoded CID characters in 52 characters (34.62%)" in caplog.text
