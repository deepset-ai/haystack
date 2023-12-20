import logging
from unittest.mock import patch
from pathlib import Path

import pytest

from haystack.dataclasses import ByteStream
from haystack.components.converters.txt import TextFileToDocument


class TestTextfileToDocument:
    def test_run_no_meta(self, test_files_path):
        """
        Test if the component runs correctly with no metadata provided separately
        """
        bytestream = ByteStream.from_file_path(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["file_path"] = str(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["key"] = "value"
        files = [str(test_files_path / "txt" / "doc_1.txt"), test_files_path / "txt" / "doc_2.txt", bytestream]
        converter = TextFileToDocument()
        output = converter.run(sources=files)
        docs = output["documents"]
        assert len(docs) == 3
        assert "Some text for testing." in docs[0].content
        assert "This is a test line." in docs[1].content
        assert "That's yet another file!" in docs[2].content
        assert docs[0].meta["file_path"] == str(files[0])
        assert docs[1].meta["file_path"] == str(files[1])
        assert docs[2].meta == bytestream.metadata

    def test_run_single_metadata_dictionary(self, test_files_path):
        """
        Test if the component runs correctly wne given a single metadata dictionary
        """
        bytestream = ByteStream.from_file_path(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["file_path"] = str(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["key"] = "value"
        files = [str(test_files_path / "txt" / "doc_1.txt"), test_files_path / "txt" / "doc_2.txt", bytestream]
        converter = TextFileToDocument()
        output = converter.run(sources=files, meta={"test-key": "test-value"})
        docs = output["documents"]
        assert len(docs) == 3
        assert "Some text for testing." in docs[0].content
        assert "This is a test line." in docs[1].content
        assert "That's yet another file!" in docs[2].content
        assert docs[0].meta == {"file_path": str(files[0]), "test-key": "test-value"}
        assert docs[1].meta == {"file_path": str(files[1]), "test-key": "test-value"}
        assert docs[2].meta == {**bytestream.metadata, "test-key": "test-value"}

    def test_run_correct_metadata_list(self, test_files_path):
        """
        Test if the component runs correctly wne given a list of metadata dictionaries of the correct length
        """
        bytestream = ByteStream.from_file_path(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["file_path"] = str(test_files_path / "txt" / "doc_3.txt")
        bytestream.metadata["key"] = "value"
        files = [str(test_files_path / "txt" / "doc_1.txt"), test_files_path / "txt" / "doc_2.txt", bytestream]
        converter = TextFileToDocument()
        output = converter.run(sources=files, meta=[{"a": "a"}, {"b": "b"}, {"c": "c"}])
        docs = output["documents"]
        assert len(docs) == 3
        assert "Some text for testing." in docs[0].content
        assert "This is a test line." in docs[1].content
        assert "That's yet another file!" in docs[2].content
        assert docs[0].meta == {"file_path": str(files[0]), "a": "a"}
        assert docs[1].meta == {"file_path": str(files[1]), "b": "b"}
        assert docs[2].meta == {**bytestream.metadata, "c": "c"}

    def test_run_metadata_list_error_handling(self, test_files_path, caplog):
        """
        Test if the component correctly handles a list of metadata of the wrong length.
        """
        paths = [test_files_path / "txt" / "doc_1.txt", test_files_path / "txt" / "doc_3.txt"]
        converter = TextFileToDocument()
        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources"):
            converter.run(sources=paths, meta=[{"a": "a"}, {"b": "b"}, {"c": "c"}])

    def test_run_error_handling(self, test_files_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = [test_files_path / "txt" / "doc_1.txt", "non_existing_file.txt", test_files_path / "txt" / "doc_3.txt"]
        converter = TextFileToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=paths)
            assert "non_existing_file.txt" in caplog.text
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].meta["file_path"] == str(paths[0])
        assert docs[1].meta["file_path"] == str(paths[2])

    def test_encoding_override(self, test_files_path):
        """
        Test if the encoding metadata field is used properly
        """
        bytestream = ByteStream.from_file_path(test_files_path / "txt" / "doc_1.txt")
        bytestream.metadata["key"] = "value"

        converter = TextFileToDocument(encoding="utf-16")
        output = converter.run(sources=[bytestream])
        assert "Some text for testing." not in output["documents"][0].content

        bytestream.metadata["encoding"] = "utf-8"
        output = converter.run(sources=[bytestream])
        assert "Some text for testing." in output["documents"][0].content

    def test_run_with_meta(self):
        bytestream = ByteStream(data=b"test", metadata={"author": "test_author", "language": "en"})

        converter = TextFileToDocument()

        output = converter.run(sources=[bytestream], meta=[{"language": "it"}])
        document = output["documents"][0]

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert document.meta == {"author": "test_author", "language": "it"}
