# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import patch
import pandas as pd
from pathlib import Path
import os

import pytest

from haystack.dataclasses import ByteStream
from haystack.components.converters.csv import CSVToDocument


@pytest.fixture
def csv_converter():
    return CSVToDocument()


class TestCSVToDocument:
    def test_init(self, csv_converter):
        assert isinstance(csv_converter, CSVToDocument)

    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        bytestream = ByteStream.from_file_path(test_files_path / "csv" / "sample_1.csv")
        bytestream.meta["file_path"] = str(test_files_path / "csv" / "sample_1.csv")
        bytestream.meta["key"] = "value"
        files = [bytestream, test_files_path / "csv" / "sample_2.csv", test_files_path / "csv" / "sample_3.csv"]
        converter = CSVToDocument()
        output = converter.run(sources=files)
        docs = output["documents"]
        assert len(docs) == 3
        assert "Name,Age\r\nJohn Doe,27\r\nJane Smith,37\r\nMike Johnson,47\r\n" == docs[0].content
        assert isinstance(docs[0].content, str)
        assert docs[0].meta == {"file_path": os.path.basename(bytestream.meta["file_path"]), "key": "value"}
        assert docs[1].meta["file_path"] == os.path.basename(files[1])
        assert docs[2].meta["file_path"] == os.path.basename(files[2])

    def test_run_with_store_full_path_false(self, test_files_path):
        """
        Test if the component runs correctly with store_full_path=False
        """
        bytestream = ByteStream.from_file_path(test_files_path / "csv" / "sample_1.csv")
        bytestream.meta["file_path"] = str(test_files_path / "csv" / "sample_1.csv")
        bytestream.meta["key"] = "value"
        files = [bytestream, test_files_path / "csv" / "sample_2.csv", test_files_path / "csv" / "sample_3.csv"]
        converter = CSVToDocument(store_full_path=False)
        output = converter.run(sources=files)
        docs = output["documents"]
        assert len(docs) == 3
        assert "Name,Age\r\nJohn Doe,27\r\nJane Smith,37\r\nMike Johnson,47\r\n" == docs[0].content
        assert isinstance(docs[0].content, str)
        assert docs[0].meta["file_path"] == "sample_1.csv"
        assert docs[0].meta["key"] == "value"
        assert docs[1].meta["file_path"] == "sample_2.csv"
        assert docs[2].meta["file_path"] == "sample_3.csv"

    def test_run_error_handling(self, test_files_path, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = [
            test_files_path / "csv" / "sample_2.csv",
            "non_existing_file.csv",
            test_files_path / "csv" / "sample_3.csv",
        ]
        converter = CSVToDocument()
        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=paths)
            assert "non_existing_file.csv" in caplog.text
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].meta["file_path"] == os.path.basename(paths[0])

    def test_encoding_override(self, test_files_path, caplog):
        """
        Test if the encoding metadata field is used properly
        """
        bytestream = ByteStream.from_file_path(test_files_path / "csv" / "sample_1.csv")
        bytestream.meta["key"] = "value"

        converter = CSVToDocument(encoding="utf-16-le")
        output = converter.run(sources=[bytestream])
        with caplog.at_level(logging.ERROR):
            output = converter.run(sources=[bytestream])
            assert "codec can't decode" in caplog.text

        converter = CSVToDocument(encoding="utf-8")
        output = converter.run(sources=[bytestream])
        assert "Name,Age\r\n" in output["documents"][0].content

    def test_run_with_meta(self):
        bytestream = ByteStream(
            data=b"Name,Age,City\r\nAlice,30,New York\r\nBob,25,Los Angeles\r\nCharlie,35,Chicago\r\n",
            meta={"name": "test_name", "language": "en"},
        )
        converter = CSVToDocument()
        output = converter.run(sources=[bytestream], meta=[{"language": "it"}])
        document = output["documents"][0]

        # check that the metadata from the bytestream is merged with that from the meta parameter
        assert document.meta == {"name": "test_name", "language": "it"}
