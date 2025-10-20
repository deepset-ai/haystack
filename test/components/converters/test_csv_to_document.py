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
        _ = converter.run(sources=[bytestream])
        with caplog.at_level(logging.ERROR):
            _ = converter.run(sources=[bytestream])
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
        assert document.meta == {"name": "test_name", "language": "it"}

    # --- NEW TESTS for strict row mode ---

    def test_row_mode_requires_content_column_param(self, tmp_path):
        # Missing content_column must raise in row mode
        f = tmp_path / "t.csv"
        f.write_text("a,b\r\n1,2\r\n", encoding="utf-8")
        conv = CSVToDocument(conversion_mode="row")
        with pytest.raises(ValueError):
            _ = conv.run(sources=[f])  # content_column missing

    def test_row_mode_missing_header_raises(self, tmp_path):
        # content_column must exist in header
        f = tmp_path / "t.csv"
        f.write_text("a,b\r\n1,2\r\n", encoding="utf-8")
        conv = CSVToDocument(conversion_mode="row")
        with pytest.raises(ValueError):
            _ = conv.run(sources=[f], content_column="missing")

    def test_row_mode_with_content_column(self, tmp_path):
        csv_text = "text,author,stars\r\nNice app,Ada,5\r\nBuggy,Bob,2\r\n"
        f = tmp_path / "fb.csv"
        f.write_text(csv_text, encoding="utf-8")

        bytestream = ByteStream.from_file_path(f)
        bytestream.meta["file_path"] = str(f)

        converter = CSVToDocument(conversion_mode="row")
        output = converter.run(sources=[bytestream], content_column="text")
        docs = output["documents"]

        assert len(docs) == 2
        assert [d.content for d in docs] == ["Nice app", "Buggy"]
        assert docs[0].meta["author"] == "Ada"
        assert docs[0].meta["stars"] == "5"
        assert docs[0].meta["row_number"] == 0
        assert os.path.basename(f) == docs[0].meta["file_path"]

    def test_row_mode_meta_collision_prefixed(self, tmp_path):
        # ByteStream meta has file_path and encoding; CSV also has those columns.
        csv_text = "file_path,encoding,comment\r\nrowpath.csv,latin1,ok\r\n"
        f = tmp_path / "collide.csv"
        f.write_text(csv_text, encoding="utf-8")
        bs = ByteStream.from_file_path(f)
        bs.meta["file_path"] = str(f)
        bs.meta["encoding"] = "utf-8"

        conv = CSVToDocument(conversion_mode="row")
        out = conv.run(sources=[bs], content_column="comment")
        d = out["documents"][0]
        # Original meta preserved
        assert d.meta["file_path"] == os.path.basename(str(f))
        assert d.meta["encoding"] == "utf-8"
        # CSV columns stored with csv_ prefix (no clobber)
        assert d.meta["csv_file_path"] == "rowpath.csv"
        assert d.meta["csv_encoding"] == "latin1"
        # content column isn't duplicated in meta
        assert "comment" not in d.meta
        assert d.meta["row_number"] == 0
        assert d.content == "ok"

    def test_row_mode_meta_collision_multiple_suffixes(self, tmp_path):
        """
        If meta already has csv_file_path and csv_file_path_1, we should write the next as csv_file_path_2.
        """
        csv_text = "file_path,comment\r\nrow.csv,ok\r\n"
        f = tmp_path / "multi.csv"
        f.write_text(csv_text, encoding="utf-8")

        bs = ByteStream.from_file_path(f)
        bs.meta["file_path"] = str(f)

        # Pre-seed meta so we force two collisions.
        extra_meta = {"csv_file_path": "existing0", "csv_file_path_1": "existing1"}

        conv = CSVToDocument(conversion_mode="row")
        out = conv.run(sources=[bs], meta=[extra_meta], content_column="comment")
        d = out["documents"][0]

        assert d.meta["csv_file_path"] == "existing0"
        assert d.meta["csv_file_path_1"] == "existing1"
        assert d.meta["csv_file_path_2"] == "row.csv"
        assert d.content == "ok"

    def test_init_validates_delimiter_and_quotechar(self):
        with pytest.raises(ValueError):
            CSVToDocument(delimiter=";;")
        with pytest.raises(ValueError):
            CSVToDocument(quotechar='""')

    def test_row_mode_large_file_warns(self, caplog, monkeypatch):
        # Make the threshold tiny so the warning always triggers.
        import haystack.components.converters.csv as csv_mod

        monkeypatch.setattr(csv_mod, "_ROW_MODE_SIZE_WARN_BYTES", 1, raising=False)

        bs = ByteStream(data=b"text,author\nhi,Ada\n", meta={"file_path": "big.csv"})
        conv = CSVToDocument(conversion_mode="row")
        with caplog.at_level(logging.WARNING, logger="haystack.components.converters.csv"):
            _ = conv.run(sources=[bs], content_column="text")
        assert "parsing a large CSV" in caplog.text

    def test_row_mode_reader_failure_raises_runtimeerror(self, monkeypatch, tmp_path):
        # Simulate DictReader failing -> we should raise RuntimeError (no fallback).
        import haystack.components.converters.csv as csv_mod

        f = tmp_path / "bad.csv"
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        conv = CSVToDocument(conversion_mode="row")

        class Boom(Exception):
            pass

        def broken_reader(*_args, **_kwargs):  # noqa: D401
            raise Boom("broken")

        monkeypatch.setattr(csv_mod.csv, "DictReader", broken_reader, raising=True)
        with pytest.raises(RuntimeError):
            _ = conv.run(sources=[f], content_column="a")
