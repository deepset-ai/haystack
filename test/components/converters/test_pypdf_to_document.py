# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import patch

import pytest

from haystack import Document, default_from_dict, default_to_dict
from haystack.components.converters.pypdf import DefaultConverter, PyPDFToDocument
from haystack.dataclasses import ByteStream


@pytest.fixture
def pypdf_converter():
    return PyPDFToDocument()


class TestPyPDFToDocument:
    def test_init(self, pypdf_converter):
        assert isinstance(pypdf_converter.converter, DefaultConverter)

    def test_to_dict(self, pypdf_converter):
        data = pypdf_converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.pypdf.PyPDFToDocument",
            "init_parameters": {
                "converter": {"type": "haystack.components.converters.pypdf.DefaultConverter", "init_parameters": {}}
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.converters.pypdf.PyPDFToDocument",
            "init_parameters": {
                "converter": {"type": "haystack.components.converters.pypdf.DefaultConverter", "init_parameters": {}}
            },
        }
        instance = PyPDFToDocument.from_dict(data)
        assert isinstance(instance, PyPDFToDocument)
        assert isinstance(instance.converter, DefaultConverter)

    @pytest.mark.integration
    def test_run(self, test_files_path, pypdf_converter):
        """
        Test if the component runs correctly.
        """
        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        output = pypdf_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "History" in docs[0].content

    @pytest.mark.integration
    def test_page_breaks_added(self, test_files_path, pypdf_converter):
        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        output = pypdf_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content.count("\f") == 3

    def test_run_with_meta(self, test_files_path, pypdf_converter):
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})

        with patch("haystack.components.converters.pypdf.PdfReader"):
            output = pypdf_converter.run(
                sources=[bytestream, test_files_path / "pdf" / "sample_pdf_1.pdf"], meta={"language": "it"}
            )

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"
        assert output["documents"][1].meta["language"] == "it"

    def test_run_error_handling(self, test_files_path, pypdf_converter, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.pdf"]
        with caplog.at_level(logging.WARNING):
            pypdf_converter.run(sources=paths)
            assert "Could not read non_existing_file.pdf" in caplog.text

    @pytest.mark.integration
    def test_mixed_sources_run(self, test_files_path, pypdf_converter):
        """
        Test if the component runs correctly when mixed sources are provided.
        """
        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        with open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb") as f:
            paths.append(ByteStream(f.read()))

        output = pypdf_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert "History and standardization" in docs[0].content
        assert "History and standardization" in docs[1].content

    @pytest.mark.integration
    def test_custom_converter(self, test_files_path):
        """
        Test if the component correctly handles custom converters.
        """
        from pypdf import PdfReader

        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]

        class MyCustomConverter:
            def convert(self, reader: PdfReader) -> Document:
                return Document(content="I don't care about converting given pdfs, I always return this")

            def to_dict(self):
                return default_to_dict(self)

            @classmethod
            def from_dict(cls, data):
                return default_from_dict(cls, data)

        component = PyPDFToDocument(converter=MyCustomConverter())
        output = component.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "ReAct" not in docs[0].content
        assert "I don't care about converting given pdfs, I always return this" in docs[0].content
