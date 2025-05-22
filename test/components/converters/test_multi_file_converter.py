# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.core.pipeline.base import component_to_dict, component_from_dict
from haystack.core.component.component import Component
from haystack.dataclasses import ByteStream
from haystack.components.converters.multi_file_converter import MultiFileConverter


@pytest.fixture
def converter():
    converter = MultiFileConverter()
    converter.warm_up()
    return converter


class TestMultiFileConverter:
    def test_init_default_params(self, converter):
        """Test initialization with default parameters"""
        assert converter.encoding == "utf-8"
        assert converter.json_content_key == "content"
        assert isinstance(converter, Component)

    def test_init_custom_params(self, converter):
        """Test initialization with custom parameters"""
        converter = MultiFileConverter(encoding="latin-1", json_content_key="text")
        assert converter.encoding == "latin-1"
        assert converter.json_content_key == "text"

    def test_to_dict(self, converter):
        """Test serialization to dictionary"""
        data = component_to_dict(converter, "converter")
        assert data == {
            "type": "haystack.components.converters.multi_file_converter.MultiFileConverter",
            "init_parameters": {"encoding": "utf-8", "json_content_key": "content"},
        }

    def test_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "type": "haystack.components.converters.multi_file_converter.MultiFileConverter",
            "init_parameters": {"encoding": "latin-1", "json_content_key": "text"},
        }
        conv = component_from_dict(MultiFileConverter, data, "converter")
        assert conv.encoding == "latin-1"
        assert conv.json_content_key == "text"

    @pytest.mark.parametrize(
        "suffix,file_path",
        [
            ("csv", "csv/sample_1.csv"),
            ("docx", "docx/sample_docx.docx"),
            ("html", "html/what_is_haystack.html"),
            ("json", "json/json_conversion_testfile.json"),
            ("md", "markdown/sample.md"),
            ("pdf", "pdf/sample_pdf_1.pdf"),
            ("pptx", "pptx/sample_pptx.pptx"),
            ("txt", "txt/doc_1.txt"),
            ("xlsx", "xlsx/table_empty_rows_and_columns.xlsx"),
        ],
    )
    @pytest.mark.integration
    def test_run(self, test_files_path, converter, suffix, file_path):
        unclassified_bytestream = ByteStream(b"unclassified content")
        unclassified_bytestream.meta["content_type"] = "unknown_type"

        paths = [test_files_path / file_path, unclassified_bytestream]

        output = converter.run(sources=paths)
        docs = output["documents"]
        unclassified = output["unclassified"]

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content is not None
        assert docs[0].meta["file_path"].endswith(suffix)

        assert len(unclassified) == 1
        assert isinstance(unclassified[0], ByteStream)
        assert unclassified[0].meta["content_type"] == "unknown_type"

    def test_run_with_meta(self, test_files_path, converter):
        """Test conversion with metadata"""
        paths = [test_files_path / "txt" / "doc_1.txt"]
        meta = {"language": "en", "author": "test"}
        output = converter.run(sources=paths, meta=meta)
        docs = output["documents"]
        assert docs[0].meta["language"] == "en"
        assert docs[0].meta["author"] == "test"

    def test_run_with_bytestream(self, test_files_path, converter):
        """Test converting ByteStream input"""
        bytestream = ByteStream(data=b"test content", mime_type="text/plain", meta={"file_path": "test.txt"})
        output = converter.run(sources=[bytestream])
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "test content"
        assert docs[0].meta["file_path"] == "test.txt"

    def test_run_error_handling(self, test_files_path, converter, caplog):
        """Test error handling for non-existent files"""
        paths = [test_files_path / "non_existent.txt"]
        with caplog.at_level("WARNING"):
            output = converter.run(sources=paths)
            assert "Could not read" in caplog.text
            assert len(output["documents"]) == 0

    @pytest.mark.integration
    def test_run_all_file_types(self, test_files_path, converter):
        """Test converting all supported file types in parallel"""
        paths = [
            test_files_path / "csv" / "sample_1.csv",
            test_files_path / "docx" / "sample_docx.docx",
            test_files_path / "html" / "what_is_haystack.html",
            test_files_path / "json" / "json_conversion_testfile.json",
            test_files_path / "markdown" / "sample.md",
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "pdf" / "sample_pdf_1.pdf",
            test_files_path / "pptx" / "sample_pptx.pptx",
            test_files_path / "xlsx" / "table_empty_rows_and_columns.xlsx",
        ]
        output = converter.run(sources=paths)
        docs = output["documents"]

        # Verify we got a document for each file
        assert len(docs) == len(paths)
        assert all(isinstance(doc, Document) for doc in docs)

    @pytest.mark.integration
    def test_run_in_pipeline(self, test_files_path, converter):
        pipeline = Pipeline(max_runs_per_component=1)
        pipeline.add_component("converter", converter)

        paths = [test_files_path / "txt" / "doc_1.txt", test_files_path / "pdf" / "sample_pdf_1.pdf"]

        output = pipeline.run(data={"sources": paths})
        docs = output["converter"]["documents"]

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.content is not None for doc in docs)


def test_import_document_preprocessor() -> None:
    # test if the MultiFileConverter.run() doesn't trigger any type or static analyzer errors
    converter = MultiFileConverter()
    converter.run(sources=[])
