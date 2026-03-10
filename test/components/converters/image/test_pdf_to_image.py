# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from haystack.components.converters.image.image_utils import _convert_pdf_to_images
from haystack.components.converters.image.pdf_to_image import PDFToImageContent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestPDFToImageContent:
    def test_to_dict(self) -> None:
        converter = PDFToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": None, "size": None, "page_range": None},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = PDFToImageContent(detail="low", size=(128, 128), page_range=[1])
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": "low", "size": (128, 128), "page_range": [1]},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {"detail": "auto", "size": None, "page_range": [1]},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }
        converter = component_from_dict(PDFToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    def test_run_with_valid_source(self) -> None:
        file_path = "./test/test_files/pdf/sample_pdf_1.pdf"
        mime_type = "application/pdf"
        converter = PDFToImageContent()
        results = converter.run(sources=[file_path])

        byte_stream = ByteStream.from_file_path(Path(file_path), mime_type=mime_type, meta={"file_path": file_path})
        assert len(results["image_contents"]) == 4
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _convert_pdf_to_images(bytestream=byte_stream, size=None, page_range=[1], return_base64=True)[0][1]
        )
        assert results["image_contents"][0].mime_type == "image/jpeg"
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == str(Path(file_path))
        assert results["image_contents"][0].meta["page_number"] == 1
        assert results["image_contents"][1].meta["page_number"] == 2
        assert results["image_contents"][2].meta["page_number"] == 3
        assert results["image_contents"][3].meta["page_number"] == 4

    def test_run_with_no_sources(self) -> None:
        converter = PDFToImageContent()
        results = converter.run(sources=[])
        assert len(results["image_contents"]) == 0
        assert results == {"image_contents": []}

    def test_run_with_invalid_source_type(self, caplog) -> None:
        converter = PDFToImageContent()
        converter.run(sources=[123])  # Invalid source type
        assert "Could not read" in caplog.text

    def test_run_with_non_existent_file(self, caplog) -> None:
        converter = PDFToImageContent()
        converter.run(sources=["./non_existent_file.png"])
        assert "Could not read" in caplog.text
        assert "No such file or directory:" in caplog.text

    def test_run_with_bytestream_sources(self) -> None:
        file_path = "./test/test_files/pdf/sample_pdf_1.pdf"
        mime_type = "application/pdf"
        byte_stream = ByteStream.from_file_path(Path(file_path), mime_type=mime_type, meta={"file_path": file_path})

        # Initialize the converter
        converter = PDFToImageContent()

        # Run the converter with the ByteStream
        results = converter.run(sources=[byte_stream])

        # Assertions
        assert len(results["image_contents"]) == 4
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _convert_pdf_to_images(bytestream=byte_stream, size=None, page_range=[1], return_base64=True)[0][1]
        )
        assert results["image_contents"][0].mime_type == "image/jpeg"
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == file_path
        assert results["image_contents"][0].meta["page_number"] == 1

    def test_run_with_empty_bytestream(self) -> None:
        # Create an empty ByteStream object
        byte_stream = ByteStream(data=b"", meta={"file_path": "empty_file.pdf"})

        # Initialize the converter
        converter = PDFToImageContent()

        # Run the converter with the empty ByteStream
        results = converter.run(sources=[byte_stream])

        assert results["image_contents"] == []
