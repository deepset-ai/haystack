import logging
import pytest

from haystack import Document
from haystack.components.converters.pypdf import PyPDFToDocument, CONVERTERS_REGISTRY
from haystack.dataclasses import ByteStream


@pytest.mark.integration
class TestPyPDFToDocument:
    def test_init(self):
        component = PyPDFToDocument()
        assert component.converter_name == "default"
        assert hasattr(component, "_converter")

    def test_init_fail_nonexisting_converter(self):
        with pytest.raises(ValueError):
            PyPDFToDocument(converter_name="non_existing_converter")

    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        paths = [test_files_path / "pdf" / "react_paper.pdf"]
        converter = PyPDFToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" in docs[0].content

    def test_run_error_handling(self, test_files_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.pdf"]
        converter = PyPDFToDocument()
        with caplog.at_level(logging.WARNING):
            converter.run(sources=paths)
            assert "Could not read non_existing_file.pdf" in caplog.text

    def test_mixed_sources_run(self, test_files_path):
        """
        Test if the component runs correctly when mixed sources are provided.
        """
        paths = [test_files_path / "pdf" / "react_paper.pdf"]
        with open(test_files_path / "pdf" / "react_paper.pdf", "rb") as f:
            paths.append(ByteStream(f.read()))

        converter = PyPDFToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert "ReAct" in docs[0].content
        assert "ReAct" in docs[1].content

    def test_custom_converter(self, test_files_path):
        """
        Test if the component correctly handles custom converters.
        """
        from pypdf import PdfReader

        paths = [test_files_path / "pdf" / "react_paper.pdf"]

        class MyCustomConverter:
            def convert(self, reader: PdfReader) -> Document:
                return Document(content="I don't care about converting given pdfs, I always return this")

        CONVERTERS_REGISTRY["custom"] = MyCustomConverter()

        converter = PyPDFToDocument(converter_name="custom")
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" not in docs[0].content
        assert "I don't care about converting given pdfs, I always return this" in docs[0].content
