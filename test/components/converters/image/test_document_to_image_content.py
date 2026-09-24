# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from PIL import Image

from haystack import Document
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestDocumentToImageContent:
    def test_to_dict(self) -> None:
        converter = DocumentToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"file_path_meta_field": "file_path", "root_path": "", "detail": None, "size": None},
            "type": "haystack.components.converters.image.document_to_image.DocumentToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = DocumentToImageContent(
            file_path_meta_field="image_path", root_path="/data", detail="high", size=(800, 600)
        )
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {
                "file_path_meta_field": "image_path",
                "root_path": "/data",
                "detail": "high",
                "size": (800, 600),
            },
            "type": "haystack.components.converters.image.document_to_image.DocumentToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {
                "file_path_meta_field": "image_path",
                "root_path": "/test",
                "detail": "auto",
                "size": (512, 512),
            },
            "type": "haystack.components.converters.image.document_to_image.DocumentToImageContent",
        }
        converter = component_from_dict(DocumentToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    def test_run_with_empty_documents_list(self) -> None:
        converter = DocumentToImageContent()
        results = converter.run(documents=[])
        assert results == {"image_contents": []}

    @pytest.mark.parametrize(
        "meta, reason",
        [
            ({}, "is missing the 'file_path' key"),
            ({"file_path": "test/test_files/docx/sample_docx.docx"}, "has an unsupported MIME type"),
            ({"file_path": "wrong_name.jpg"}, "has an invalid file path 'wrong_name.jpg'"),
            ({"file_path": "test/test_files/pdf/sample_pdf_1.pdf"}, "is missing the 'page_number' key"),
        ],
    )
    def test_run_with_invalid_document(self, meta: dict, reason: str, caplog: pytest.LogCaptureFixture) -> None:
        converter = DocumentToImageContent()
        results = converter.run(documents=[Document(content="test", meta=meta)])
        assert results == {"image_contents": [None]}
        assert reason in caplog.text

    def test_run_with_image_documents(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files/images")
        image_doc = Document(content="test", meta={"file_path": "apple.jpg"})
        results = converter.run(documents=[image_doc])
        assert len(results["image_contents"]) == 1
        image_content = results["image_contents"][0]
        assert image_content is not None
        assert image_content.meta == {"file_path": "apple.jpg"}

    def test_run_with_pdf_documents(self) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1})
        results = converter.run(documents=[pdf_doc])
        assert len(results["image_contents"]) == 1
        image_content = results["image_contents"][0]
        assert image_content is not None
        assert image_content.meta == {"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1}

    def test_run_with_mixed_document_types(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files")
        documents = [
            Document(content="", meta={"file_path": "images/apple.jpg"}),
            Document(content="", meta={"file_path": "pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="text", meta={"file_path": "docx/sample_docx.docx"}),
        ]
        image_contents = converter.run(documents=documents)["image_contents"]
        assert image_contents[0] is not None
        assert image_contents[1] is not None
        assert image_contents[2] is None

    @patch("haystack.components.converters.image.document_to_image._extract_image_sources_info")
    @patch("haystack.components.converters.image.document_to_image._batch_convert_pdf_pages_to_images")
    @patch("PIL.Image.open")
    @patch("haystack.components.converters.image.document_to_image.ByteStream")
    def test_run_none_images(
        self,
        mocked_byte_stream,
        mocked_pil_open,
        mocked_batch_convert_pdf_pages_to_images,
        mocked_extract_image_sources_info,
        caplog,
    ):
        converter = DocumentToImageContent()

        mocked_extract_image_sources_info.side_effect = [
            [{"path": "doc1.pdf", "mime_type": "application/pdf", "page_number": 999}],  # Page 999 doesn't exist
            [{"path": "image1.jpg", "mime_type": "image/jpeg"}],
        ]
        mocked_batch_convert_pdf_pages_to_images.return_value = {}  # Empty dict because page was skipped
        mocked_pil_open.return_value = Image.new("RGB", (100, 100))
        mocked_byte_stream.from_file_path.return_value = ByteStream(b"")

        documents = [
            Document(content="PDF 1", meta={"file_path": "doc1.pdf", "page_number": 999}),
            Document(content="Image 1", meta={"file_path": "image1.jpg"}),
        ]

        image_contents = converter.run(documents=documents)["image_contents"]

        assert caplog.records[-1].levelname == "WARNING"
        assert "Conversion failed for some documents." in caplog.records[-1].message

        assert image_contents[0] is None
        assert image_contents[1] is not None
