import logging
import datetime

import pytest

from haystack.dataclasses import ByteStream
from haystack.components.converters.docx import DocxToDocument, DocxMetadata


@pytest.fixture
def docx_converter():
    return DocxToDocument()


class TestDocxToDocument:
    def test_init(self, docx_converter):
        assert isinstance(docx_converter, DocxToDocument)

    def test_run(self, test_files_path, docx_converter):
        """
        Test if the component runs correctly
        """
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "History" in docs[0].content
        assert docs[0].meta.keys() == {"file_path", "docx"}
        assert docs[0].meta == {
            "file_path": str(paths[0]),
            "docx": DocxMetadata(
                author="Microsoft Office User",
                category="",
                comments="",
                content_status="",
                created=datetime.datetime(2024, 6, 9, 21, 17, tzinfo=datetime.timezone.utc),
                identifier="",
                keywords="",
                language="",
                last_modified_by="Carlos Fernández Lorán",
                last_printed=None,
                modified=datetime.datetime(2024, 6, 9, 21, 27, tzinfo=datetime.timezone.utc),
                revision=2,
                subject="",
                title="",
                version="",
            ),
        }

    def test_run_with_meta_overwrites(self, test_files_path, docx_converter):
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths, meta={"language": "it", "author": "test_author"})
        doc = output["documents"][0]
        assert doc.meta == {
            "file_path": str(paths[0]),
            "docx": DocxMetadata(
                author="Microsoft Office User",
                category="",
                comments="",
                content_status="",
                created=datetime.datetime(2024, 6, 9, 21, 17, tzinfo=datetime.timezone.utc),
                identifier="",
                keywords="",
                language="",
                last_modified_by="Carlos Fernández Lorán",
                last_printed=None,
                modified=datetime.datetime(2024, 6, 9, 21, 27, tzinfo=datetime.timezone.utc),
                revision=2,
                subject="",
                title="",
                version="",
            ),
            "language": "it",
            "author": "test_author",
        }

    def test_run_error_wrong_file_type(self, caplog, test_files_path, docx_converter):
        sources = [str(test_files_path / "txt" / "doc_1.txt")]
        with caplog.at_level(logging.WARNING):
            results = docx_converter.run(sources=sources)
            assert "doc_1.txt and convert it" in caplog.text
            assert results["documents"] == []

    def test_run_error_non_existent_file(self, test_files_path, docx_converter, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.docx"]
        with caplog.at_level(logging.WARNING):
            docx_converter.run(sources=paths)
            assert "Could not read non_existing_file.docx" in caplog.text

    def test_mixed_sources_run(self, test_files_path, docx_converter):
        """
        Test if the component runs correctly when mixed sources are provided.
        """
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        with open(test_files_path / "docx" / "sample_docx_1.docx", "rb") as f:
            paths.append(ByteStream(f.read()))

        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 2
        assert "History and standardization" in docs[0].content
        assert "History and standardization" in docs[1].content
