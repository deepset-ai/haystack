import logging
from unittest.mock import patch
from pathlib import Path

import pytest

from haystack.preview.dataclasses import ByteStream
from haystack.preview.components.file_converters.txt import TextFileToDocument


class TestTextfileToDocument:
    @pytest.mark.unit
    def test_run(self, preview_samples_path):
        """
        Test if the component runs correctly.
        """
        bytestream = ByteStream.from_file_path(preview_samples_path / "txt" / "doc_3.txt")
        bytestream.metadata["file_path"] = str(preview_samples_path / "txt" / "doc_3.txt")
        files = [
            str(preview_samples_path / "txt" / "doc_1.txt"),
            preview_samples_path / "txt" / "doc_2.txt",
            bytestream,
        ]
        converter = TextFileToDocument()
        output = converter.run(streams=files)
        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content == "Some text for testing.\nTwo lines in here."
        assert docs[1].content == "This is a test line.\n123 456 789\n987 654 321."
        assert docs[1].content == "That's yet another file!\n\nit contains\n\n\n\nmany\n\nempty lines."
        assert docs[0].meta["file_path"] == str(files[0])
        assert docs[1].meta["file_path"] == str(files[1])
        assert docs[1].meta["file_path"] == str(files[2])

    @pytest.mark.unit
    def test_run_error_handling(self, preview_samples_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = [
            preview_samples_path / "txt" / "doc_1.txt",
            "non_existing_file.txt",
            preview_samples_path / "txt" / "doc_3.txt",
        ]
        converter = TextFileToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(paths=paths)
            assert (
                "Could not read file non_existing_file.txt. Skipping it. Error message: File at path non_existing_file.txt does not exist."
                in caplog.text
            )
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].meta["file_path"] == str(paths[0])
        assert docs[1].meta["file_path"] == str(paths[2])
