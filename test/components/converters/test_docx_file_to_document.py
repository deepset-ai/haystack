import logging
from unittest.mock import patch

import pytest

from haystack.dataclasses import ByteStream
from haystack.components.converters import DocxToDocument


@pytest.fixture
def docx_converter():
    return DocxToDocument()


class TestDocxToDocument:
    def test_init(self, docx_converter):
        assert isinstance(docx_converter, DocxToDocument)

    @pytest.mark.integration
    def test_run(self, test_files_path, docx_converter):
        """
        Test if the component runs correctly
        """
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "History" in docs[0].content

    def test_run_with_meta(self, test_files_path, docx_converter):
        with patch("haystack.components.converters.docx.DocxToDocument"):
            output = docx_converter.run(
                sources=[test_files_path / "docx" / "sample_docx_1.docx"],
                meta={"language": "it", "author": "test_author"},
            )

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"

    def test_run_error_handling(self, test_files_path, docx_converter, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.docx"]
        with caplog.at_level(logging.WARNING):
            docx_converter.run(sources=paths)
            assert "Could not read non_existing_file.docx" in caplog.text

    @pytest.mark.integration
    def test_mixed_sources_run(self, test_files_path, docx_converter):
        """
        Test if the component runs correctly when mixed sources are provided.
        """
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        with open(test_files_path / "docx" / "sample_docx_1.docx", "rb") as f:
            paths.append(ByteStream(f.read()))

        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert "History and standardization" in docs[0].content
        assert "History and standardization" in docs[1].content
