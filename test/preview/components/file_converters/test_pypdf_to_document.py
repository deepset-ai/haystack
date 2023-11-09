import logging
import pytest
from pypdf import PdfReader

from haystack.preview import Document
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack.preview.dataclasses import ByteStream


class TestPyPDFToDocument:
    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        paths = [preview_samples_path / "pdf" / "react_paper.pdf"]
        converter = PyPDFToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" in docs[0].content

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.pdf"]
        converter = PyPDFToDocument()
        with caplog.at_level(logging.WARNING):
            converter.run(sources=paths)
            assert "Could not read non_existing_file.pdf" in caplog.text

    @pytest.mark.unit
    def test_mixed_sources_run(self, preview_samples_path):
        """
        Test if the component runs correctly when mixed sources are provided.
        """
        paths = [preview_samples_path / "pdf" / "react_paper.pdf"]
        with open(preview_samples_path / "pdf" / "react_paper.pdf", "rb") as f:
            paths.append(ByteStream(f.read()))

        converter = PyPDFToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert "ReAct" in docs[0].content
        assert "ReAct" in docs[1].content

    @pytest.mark.unit
    def test_custom_converter(self, preview_samples_path):
        """
        Test if the component correctly handles custom converters.
        """
        paths = [preview_samples_path / "pdf" / "react_paper.pdf"]

        class MyCustomConverter:
            def convert(self, reader: PdfReader) -> Document:
                return Document(content="I don't care about converting given pdfs, I always return this")

        converter = PyPDFToDocument(converter=MyCustomConverter())
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" not in docs[0].content
        assert "I don't care about converting given pdfs, I always return this" in docs[0].content
