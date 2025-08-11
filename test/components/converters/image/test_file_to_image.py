# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.components.converters.image.file_to_image import ImageFileToImageContent
from haystack.components.converters.image.image_utils import _encode_image_to_base64
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestImageFileToImageContent:
    def test_to_dict(self) -> None:
        converter = ImageFileToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": None, "size": None},
            "type": "haystack.components.converters.image.file_to_image.ImageFileToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = ImageFileToImageContent(detail="low", size=(128, 128))
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": "low", "size": (128, 128)},
            "type": "haystack.components.converters.image.file_to_image.ImageFileToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {"detail": "auto", "size": None},
            "type": "haystack.components.converters.image.file_to_image.ImageFileToImageContent",
        }
        converter = component_from_dict(ImageFileToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    @pytest.mark.parametrize(
        ("image_path", "mime_type"),
        [
            ("./test/test_files/images/haystack-logo.png", "image/png"),
            ("./test/test_files/images/apple.jpg", "image/jpeg"),
        ],
    )
    def test_run_with_valid_sources(self, image_path: str, mime_type: str) -> None:
        converter = ImageFileToImageContent()
        results = converter.run(sources=[image_path], size=(128, 128))

        byte_stream = ByteStream.from_file_path(
            Path(image_path), mime_type=mime_type, meta={"file_name": image_path.split("/")[-1]}
        )
        assert len(results["image_contents"]) == 1
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _encode_image_to_base64(bytestream=byte_stream, size=(128, 128))[1]
        )
        assert results["image_contents"][0].mime_type == mime_type
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == str(Path(image_path))

    def test_run_with_no_sources(self) -> None:
        converter = ImageFileToImageContent()
        results = converter.run(sources=[])
        assert len(results["image_contents"]) == 0
        assert results == {"image_contents": []}

    def test_run_with_invalid_source_type(self, caplog) -> None:
        converter = ImageFileToImageContent()
        converter.run(sources=[123])  # Invalid source type
        assert "Could not read" in caplog.text

    def test_run_with_non_existent_file(self, caplog) -> None:
        converter = ImageFileToImageContent()
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

        # Initialize the converter
        converter = ImageFileToImageContent(size=(128, 128))

        # Run the converter with the ByteStream
        results = converter.run(sources=[byte_stream])

        # Assertions
        assert len(results["image_contents"]) == 1
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _encode_image_to_base64(bytestream=byte_stream, size=(128, 128))[1]
        )
        assert results["image_contents"][0].mime_type == mime_type
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == image_path

    def test_run_with_empty_bytestream(self) -> None:
        # Create an empty ByteStream object
        byte_stream = ByteStream(data=b"", meta={"file_path": "empty_file.png"})

        # Initialize the converter
        converter = ImageFileToImageContent()

        # Run the converter with the empty ByteStream
        results = converter.run(sources=[byte_stream])

        assert results["image_contents"] == []
