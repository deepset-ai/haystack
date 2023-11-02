import logging

import pytest

from haystack.preview.components.file_converters.markdown import MarkdownToDocument
from haystack.preview.dataclasses import ByteStream


class TestMarkdownToDocument:
    @pytest.mark.unit
    def test_init_params_default(self):
        converter = MarkdownToDocument()
        assert converter.table_to_single_line is False
        assert converter.progress_bar is True

    @pytest.mark.unit
    def test_init_params_custom(self):
        converter = MarkdownToDocument(table_to_single_line=True, progress_bar=False)
        assert converter.table_to_single_line is True
        assert converter.progress_bar is False

    @pytest.mark.integration
    def test_run(self, preview_samples_path):
        converter = MarkdownToDocument()
        sources = [preview_samples_path / "markdown" / "sample.md"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "What to build with Haystack" in doc.content
            assert "# git clone https://github.com/deepset-ai/haystack.git" in doc.content

    @pytest.mark.integration
    def test_run_metadata(self, preview_samples_path):
        converter = MarkdownToDocument()
        sources = [preview_samples_path / "markdown" / "sample.md"]
        metadata = [{"file_name": "sample.md"}]
        results = converter.run(sources=sources, meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "What to build with Haystack" in doc.content
            assert "# git clone https://github.com/deepset-ai/haystack.git" in doc.content
            assert doc.meta == {"file_name": "sample.md"}

    @pytest.mark.integration
    def test_run_wrong_file_type(self, preview_samples_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [preview_samples_path / "audio" / "answer.wav"]
        converter = MarkdownToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=sources)
            assert "codec can't decode byte" in caplog.text

        docs = output["documents"]
        assert not docs

    @pytest.mark.integration
    def test_run_error_handling(self, caplog):
        """
        Test if the component correctly handles errors.
        """
        sources = ["non_existing_file.md"]
        converter = MarkdownToDocument()
        with caplog.at_level(logging.WARNING):
            result = converter.run(sources=sources)
            assert "Could not read non_existing_file.md" in caplog.text
            assert not result["documents"]

    @pytest.mark.unit
    def test_mixed_sources_run(self, preview_samples_path):
        """
        Test if the component runs correctly if the input is a mix of strings, paths and ByteStreams.
        """
        sources = [
            preview_samples_path / "markdown" / "sample.md",
            str((preview_samples_path / "markdown" / "sample.md").absolute()),
        ]
        with open(preview_samples_path / "markdown" / "sample.md", "rb") as f:
            byte_stream = f.read()
            sources.append(ByteStream(byte_stream))

        converter = MarkdownToDocument()
        output = converter.run(sources=sources)
        docs = output["documents"]
        assert len(docs) == 3
        for doc in docs:
            assert "What to build with Haystack" in doc.content
            assert "# git clone https://github.com/deepset-ai/haystack.git" in doc.content
