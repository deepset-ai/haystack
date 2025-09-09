# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest

from haystack.components.converters.csv import CSVToDocument
from haystack.dataclasses import ByteStream


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
        assert docs[0].content == "Name,Age\r\nJohn Doe,27\r\nJane Smith,37\r\nMike Johnson,47\r\n"
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
        assert docs[0].content == "Name,Age\r\nJohn Doe,27\r\nJane Smith,37\r\nMike Johnson,47\r\n"
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

    def test_row_mode_with_content_column(self, tmp_path):
        """
        Each row becomes a Document, with `content` from a chosen column and other columns in meta.
        """
        csv_text = "text,author,stars\r\nNice app,Ada,5\r\nBuggy,Bob,2\r\n"
        f = tmp_path / "fb.csv"
        f.write_text(csv_text, encoding="utf-8")

        bytestream = ByteStream.from_file_path(f)
        bytestream.meta["file_path"] = str(f)

        converter = CSVToDocument(conversion_mode="row", content_column="text")
        output = converter.run(sources=[bytestream])
        docs = output["documents"]

        assert len(docs) == 2
        assert [d.content for d in docs] == ["Nice app", "Buggy"]
        # Remaining columns land in meta, plus file-level meta preserved
        assert docs[0].meta["author"] == "Ada"
        assert docs[0].meta["stars"] == "5"
        assert docs[0].meta["row_number"] == 0
        # still respects store_full_path default=False trimming when present
        assert os.path.basename(f) == docs[0].meta["file_path"]

    def test_row_mode_without_content_column(self, tmp_path):
        """
        Without `content_column`, the content is a human-readable 'key: value' listing of the row.
        """
        csv_text = "a,b\r\n1,2\r\n3,4\r\n"
        f = tmp_path / "t.csv"
        f.write_text(csv_text, encoding="utf-8")

        converter = CSVToDocument(conversion_mode="row")
        output = converter.run(sources=[f])
        docs = output["documents"]

        assert len(docs) == 2
        assert "a: 1" in docs[0].content and "b: 2" in docs[0].content
        assert docs[0].meta["a"] == "1" and docs[0].meta["b"] == "2"
        assert docs[0].meta["row_number"] == 0

    def test_row_mode_meta_merging(self, tmp_path):
        """
        File-level meta and explicit `meta` arg are merged into each row's meta.
        """
        csv_text = "q,user\r\nHello,u1\r\nHi,u2\r\n"
        f = tmp_path / "m.csv"
        f.write_text(csv_text, encoding="utf-8")

        bs = ByteStream.from_file_path(f)
        bs.meta["dataset"] = "support_tickets"

        converter = CSVToDocument(conversion_mode="row", content_column="q")
        out = converter.run(sources=[bs], meta=[{"lang": "en"}])
        docs = out["documents"]

        assert len(docs) == 2
        assert docs[0].content == "Hello"
        # merged meta propagated to each row
        assert docs[0].meta["dataset"] == "support_tickets"
        assert docs[0].meta["lang"] == "en"
        # remaining column captured
        assert docs[0].meta["user"] == "u1"
