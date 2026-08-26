# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from PIL import Image

from haystack import Document
from haystack.components.converters.image.document_to_image import DocumentToImageContent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream, ImageContent


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

    def test_run_with_missing_file_path_metadata(self, caplog) -> None:
        converter = DocumentToImageContent()
        # Document without file_path in metadata
        doc_no_path = Document(content="test", meta={})
        # Document with file_path but file doesn't exist
        doc_no_file = Document(content="test", meta={"file_path": "nonexistent.jpg"})
        results = converter.run(documents=[doc_no_path, doc_no_file])
        assert results["image_contents"] == [None, None]
        assert any(
            "Conversion failed for some documents." in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_run_with_non_image_documents(self, caplog) -> None:
        converter = DocumentToImageContent()
        docx_doc = Document(content="test", meta={"file_path": "test/test_files/docx/sample_docx.docx"})
        results = converter.run(documents=[docx_doc])
        assert results["image_contents"] == [None]
        assert any(
            "unsupported MIME type" in record.message for record in caplog.records if record.levelname == "WARNING"
        )

    def test_run_with_invalid_file_path(self, caplog) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "wrong_name.jpg"})
        results = converter.run(documents=[pdf_doc])
        assert results["image_contents"] == [None]
        assert any("has an invalid file path" in record.message for record in caplog.records)

    def test_run_with_pdf_missing_page_number(self, caplog) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf"})
        results = converter.run(documents=[pdf_doc])
        assert results["image_contents"] == [None]
        assert any("missing the 'page_number' key" in record.message for record in caplog.records)

    def test_run_with_image_documents(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files/images")
        image_doc = Document(content="test", meta={"file_path": "apple.jpg"})
        results = converter.run(documents=[image_doc])
        assert len(results["image_contents"]) == 1
        assert results["image_contents"][0].meta == {"file_path": "apple.jpg"}

    def test_run_with_pdf_documents(self) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1})
        results = converter.run(documents=[pdf_doc])
        assert len(results["image_contents"]) == 1
        assert results["image_contents"][0].meta == {
            "file_path": "test/test_files/pdf/sample_pdf_1.pdf",
            "page_number": 1,
        }

    def test_run_with_mixed_document_types(self, caplog) -> None:
        converter = DocumentToImageContent(root_path="test/test_files")
        documents = [
            Document(content="", meta={"file_path": "images/apple.jpg"}),
            Document(content="", meta={"file_path": "pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="text", meta={"file_path": "docx/sample_docx.docx"}),
        ]
        image_contents = converter.run(documents=documents)["image_contents"]
        assert isinstance(image_contents[0], ImageContent)
        assert isinstance(image_contents[1], ImageContent)
        assert image_contents[2] is None
        assert any("Conversion failed for some documents." in record.message for record in caplog.records)

    def test_run_with_mixed_valid_and_unsupported_mime_documents(self, caplog) -> None:
        converter = DocumentToImageContent(root_path="test/test_files")
        documents = [
            Document(content="", meta={"file_path": "images/apple.jpg"}),
            Document(content="", meta={"file_path": "pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="", meta={"file_path": "docx/sample_docx.docx"}),
        ]
        image_contents = converter.run(documents=documents)["image_contents"]

        assert len(image_contents) == 3
        assert isinstance(image_contents[0], ImageContent)
        assert isinstance(image_contents[1], ImageContent)
        assert image_contents[2] is None

        warning_records = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_records) == 1
        assert "Conversion failed for some documents." in warning_records[0].message
        assert "unsupported MIME type" in warning_records[0].message

    def test_run_with_mixed_pdf_documents_missing_page_number(self, caplog) -> None:
        converter = DocumentToImageContent()
        documents = [
            Document(content="", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf"}),
        ]
        image_contents = converter.run(documents=documents)["image_contents"]

        assert len(image_contents) == 2
        assert isinstance(image_contents[0], ImageContent)
        assert image_contents[1] is None

        warning_records = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_records) == 1
        assert "missing the 'page_number' key" in warning_records[0].message

    def test_run_with_out_of_range_pdf_page_returns_none(self, caplog) -> None:
        converter = DocumentToImageContent()
        doc = Document(content="", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 999})
        result = converter.run(documents=[doc])
        assert result == {"image_contents": [None]}
        assert any(
            "Conversion failed for some documents." in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )

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

        # one call per document, each returning that document's source info
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
