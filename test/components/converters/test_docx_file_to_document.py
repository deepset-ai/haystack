import logging
import datetime

import pytest

from haystack.dataclasses import ByteStream
from haystack.components.converters import DocxToDocument


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
        assert docs[0].meta.keys() == {
            "file_path",
            "author",
            "category",
            "comments",
            "content_status",
            "created",
            "identifier",
            "keywords",
            "language",
            "last_modified_by",
            "last_printed",
            "modified",
            "revision",
            "subject",
            "title",
            "version",
        }
        assert docs[0].meta == {
            "file_path": str(paths[0]),
            "author": "Microsoft Office User",
            "category": "",
            "comments": "",
            "content_status": "",
            "created": datetime.datetime(2024, 6, 9, 21, 17, tzinfo=datetime.timezone.utc),
            "identifier": "",
            "keywords": "",
            "language": "",
            "last_modified_by": "Carlos Fernández Lorán",
            "last_printed": None,
            "modified": datetime.datetime(2024, 6, 9, 21, 27, tzinfo=datetime.timezone.utc),
            "revision": 2,
            "subject": "",
            "title": "",
            "version": "",
        }

    def test_run_with_meta_overwrites(self, test_files_path, docx_converter):
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths, meta={"language": "it", "author": "test_author"})
        doc = output["documents"][0]
        assert doc.meta == {
            "file_path": str(paths[0]),
            "category": "",
            "comments": "",
            "content_status": "",
            "created": datetime.datetime(2024, 6, 9, 21, 17, tzinfo=datetime.timezone.utc),
            "identifier": "",
            "keywords": "",
            "last_modified_by": "Carlos Fernández Lorán",
            "last_printed": None,
            "modified": datetime.datetime(2024, 6, 9, 21, 27, tzinfo=datetime.timezone.utc),
            "revision": 2,
            "subject": "",
            "title": "",
            "version": "",
            "language": "it",  # This overwrites the language from the docx metadata
            "author": "test_author",  # This overwrites the author from the docx metadata
        }

    def test_run_error_handling(self, test_files_path, docx_converter, caplog):
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
