# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.components.converters.image.file_to_document import ImageFileToDocument
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestImageFileToDocument:
    def test_to_dict(self) -> None:
        converter = ImageFileToDocument()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"store_full_path": False},
            "type": "haystack.components.converters.image.file_to_document.ImageFileToDocument",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = ImageFileToDocument(store_full_path=True)
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"store_full_path": True},
            "type": "haystack.components.converters.image.file_to_document.ImageFileToDocument",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {"store_full_path": False},
            "type": "haystack.components.converters.image.file_to_document.ImageFileToDocument",
        }
        converter = component_from_dict(ImageFileToDocument, data, "name")
        assert component_to_dict(converter, "converter") == data

    @pytest.mark.parametrize(
        ("image_path", "mime_type"),
        [
            ("./test/test_files/images/haystack-logo.png", "image/png"),
            ("./test/test_files/images/apple.jpg", "image/jpeg"),
        ],
    )
    def test_run_with_valid_sources(self, image_path: str, mime_type: str) -> None:
        converter = ImageFileToDocument(store_full_path=True)
        results = converter.run(sources=[image_path], meta={"source": "test_source"})

        assert len(results["documents"]) == 1
        assert results["documents"][0].content is None
        assert results["documents"][0].meta == {"source": "test_source", "file_path": image_path}

    def test_run_with_no_sources(self) -> None:
        converter = ImageFileToDocument()
        results = converter.run(sources=[])
        assert len(results["documents"]) == 0
        assert results == {"documents": []}

    def test_run_with_invalid_source_type(self, caplog) -> None:
        converter = ImageFileToDocument()
        converter.run(sources=[123])  # Invalid source type
        assert "Could not read" in caplog.text

    def test_run_with_non_existent_file(self, caplog) -> None:
        converter = ImageFileToDocument()
        converter.run(sources=["./non_existent_file.png"])
        assert "Could not read" in caplog.text
        assert "No such file or directory:" in caplog.text

    @pytest.mark.parametrize(
        ("image_path", "mime_type"),
        [
            ("./test/test_files/images/haystack-logo.png", "image/png"),
            ("./test/test_files/images/apple.jpg", "image/jpeg"),
        ],
    )
    def test_run_with_bytestream_sources(self, image_path: str, mime_type: str) -> None:
        byte_stream = ByteStream.from_file_path(Path(image_path), mime_type=mime_type, meta={"file_path": image_path})
        converter = ImageFileToDocument(store_full_path=True)
        results = converter.run(sources=[byte_stream])
        assert len(results["documents"]) == 1
        assert results["documents"][0].content is None
        assert results["documents"][0].meta == {"file_path": image_path}
