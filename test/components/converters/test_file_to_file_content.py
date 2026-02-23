# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path

import pytest

from haystack.components.converters.file_to_file_content import FileToFileContent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestFileToFileContent:
    def test_to_dict(self) -> None:
        converter = FileToFileContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {},
            "type": "haystack.components.converters.file_to_file_content.FileToFileContent",
        }

    def test_from_dict(self) -> None:
        data = {"init_parameters": {}, "type": "haystack.components.converters.file_to_file_content.FileToFileContent"}
        converter = component_from_dict(FileToFileContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    @pytest.mark.parametrize(
        ("file_path", "mime_type"),
        [
            ("./test/test_files/pdf/sample_pdf_1.pdf", "application/pdf"),
            ("./test/test_files/txt/doc_1.txt", "text/plain"),
        ],
    )
    def test_run_with_valid_sources(self, file_path: str, mime_type: str) -> None:
        converter = FileToFileContent()
        results = converter.run(sources=[file_path])

        assert len(results["file_contents"]) == 1
        file_content = results["file_contents"][0]

        assert file_content.base64_data is not None
        assert file_content.mime_type == mime_type
        assert file_content.filename == Path(file_path).name
        assert file_content.extra == {}

        with open(file_path, "rb") as f:
            expected_base64 = base64.b64encode(f.read()).decode("utf-8")
        assert file_content.base64_data == expected_base64

    @pytest.mark.parametrize(
        ("file_path", "mime_type"),
        [
            ("./test/test_files/pdf/sample_pdf_1.pdf", "application/pdf"),
            ("./test/test_files/audio/answer.wav", "audio/x-wav"),
        ],
    )
    def test_run_with_bytestream_sources(self, file_path: str, mime_type: str) -> None:
        byte_stream = ByteStream.from_file_path(Path(file_path))

        converter = FileToFileContent()
        results = converter.run(sources=[byte_stream])

        assert len(results["file_contents"]) == 1
        file_content = results["file_contents"][0]

        assert file_content.base64_data is not None
        assert file_content.mime_type == mime_type
        assert file_content.filename is None
        assert file_content.extra == {}

        expected_base64 = base64.b64encode(byte_stream.data).decode("utf-8")
        assert file_content.base64_data == expected_base64

    def test_run_with_no_sources(self) -> None:
        converter = FileToFileContent()
        results = converter.run(sources=[])
        assert results == {"file_contents": []}

    def test_run_with_invalid_source_type(self, caplog) -> None:
        converter = FileToFileContent()
        converter.run(sources=[123])
        assert "Could not read" in caplog.text

    def test_run_with_non_existent_file(self, caplog) -> None:
        converter = FileToFileContent()
        converter.run(sources=["./non_existent_file.pdf"])
        assert "Could not read" in caplog.text

    def test_run_with_empty_bytestream(self) -> None:
        byte_stream = ByteStream(data=b"")

        converter = FileToFileContent()
        results = converter.run(sources=[byte_stream])

        assert results["file_contents"] == []

    def test_run_with_extra_dict(self) -> None:
        converter = FileToFileContent()
        extra = {"key": "value"}
        results = converter.run(sources=["./test/test_files/txt/doc_1.txt"], extra=extra)

        assert len(results["file_contents"]) == 1
        assert results["file_contents"][0].extra == extra

    def test_run_with_extra_list(self) -> None:
        converter = FileToFileContent()
        sources = ["./test/test_files/txt/doc_1.txt", "./test/test_files/txt/doc_2.txt"]
        extra = [{"key": "value1"}, {"key": "value2"}]
        results = converter.run(sources=sources, extra=extra)

        assert len(results["file_contents"]) == 2
        assert results["file_contents"][0].extra == {"key": "value1"}
        assert results["file_contents"][1].extra == {"key": "value2"}

    def test_run_skips_empty_files_among_valid(self, caplog) -> None:
        byte_stream_empty = ByteStream(data=b"")
        valid_source = "./test/test_files/txt/doc_1.txt"

        converter = FileToFileContent()
        results = converter.run(sources=[byte_stream_empty, valid_source])

        assert len(results["file_contents"]) == 1
        assert "empty" in caplog.text
