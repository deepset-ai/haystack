import logging

from unittest.mock import patch
import pytest

from haystack.components.converters.markdown import MarkdownToDocument
from haystack.dataclasses import ByteStream


@pytest.mark.integration
class TestMarkdownToDocument:
    def test_init_params_default(self):
        converter = MarkdownToDocument()
        assert converter.table_to_single_line is False
        assert converter.progress_bar is True

    def test_init_params_custom(self):
        converter = MarkdownToDocument(table_to_single_line=True, progress_bar=False)
        assert converter.table_to_single_line is True
        assert converter.progress_bar is False

    @pytest.mark.integration
    def test_run(self, test_files_path):
        converter = MarkdownToDocument()
        sources = [test_files_path / "markdown" / "sample.md"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "What to build with Haystack" in doc.content
            assert "# git clone https://github.com/deepset-ai/haystack.git" in doc.content

    def test_run_calls_normalize_metadata(self, test_files_path):
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})

        converter = MarkdownToDocument()

        with patch("haystack.components.converters.markdown.normalize_metadata") as normalize_metadata, patch(
            "haystack.components.converters.markdown.MarkdownIt"
        ):
            converter.run(sources=[bytestream, test_files_path / "markdown" / "sample.md"], meta={"language": "it"})

        # check that the metadata normalizer is called properly
        normalize_metadata.assert_called_with(meta={"language": "it"}, sources_count=2)

    def test_run_with_meta(self, test_files_path):
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})

        converter = MarkdownToDocument()

        with patch("haystack.components.converters.markdown.MarkdownIt"):
            output = converter.run(
                sources=[bytestream, test_files_path / "markdown" / "sample.md"], meta={"language": "it"}
            )

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"
        assert output["documents"][1].meta["language"] == "it"

    @pytest.mark.integration
    def test_run_wrong_file_type(self, test_files_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [test_files_path / "audio" / "answer.wav"]
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

    def test_mixed_sources_run(self, test_files_path):
        """
        Test if the component runs correctly if the input is a mix of strings, paths and ByteStreams.
        """
        sources = [
            test_files_path / "markdown" / "sample.md",
            str((test_files_path / "markdown" / "sample.md").absolute()),
        ]
        with open(test_files_path / "markdown" / "sample.md", "rb") as f:
            byte_stream = f.read()
            sources.append(ByteStream(byte_stream))

        converter = MarkdownToDocument()
        output = converter.run(sources=sources)
        docs = output["documents"]
        assert len(docs) == 3
        for doc in docs:
            assert "What to build with Haystack" in doc.content
            assert "# git clone https://github.com/deepset-ai/haystack.git" in doc.content
