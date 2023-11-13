import logging

import pytest

from haystack.preview.components.file_converters import HTMLToDocument
from haystack.preview.dataclasses import ByteStream


class TestHTMLToDocument:
    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        paths = [preview_samples_path / "html" / "what_is_haystack.html"]
        converter = HTMLToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "Haystack" in docs[0].content

    @pytest.mark.unit
    def test_run_wrong_file_type(self, preview_samples_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        paths = [preview_samples_path / "audio" / "answer.wav"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=paths)
            assert "codec can't decode byte" in caplog.text

        docs = output["documents"]
        assert not docs

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.html"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            result = converter.run(sources=paths)
            assert "Could not read non_existing_file.html" in caplog.text
            assert not result["documents"]

    @pytest.mark.unit
    def test_mixed_sources_run(self, preview_samples_path):
        """
        Test if the component runs correctly if the input is a mix of paths and ByteStreams
        """
        paths = [preview_samples_path / "html" / "what_is_haystack.html"]
        with open(preview_samples_path / "html" / "what_is_haystack.html", "rb") as f:
            byte_stream = f.read()
            paths.append(ByteStream(byte_stream))

        converter = HTMLToDocument()
        output = converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        for doc in docs:
            assert "Haystack" in doc.content
