import json
import logging

import pytest

from haystack import Document
from haystack.components.converters.docx import DOCXMetadata, DOCXToDocument
from haystack.dataclasses import ByteStream


@pytest.fixture
def docx_converter():
    return DOCXToDocument()


class TestDOCXToDocument:
    def test_init(self, docx_converter):
        assert isinstance(docx_converter, DOCXToDocument)

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
            "docx": DOCXMetadata(
                author="Microsoft Office User",
                category="",
                comments="",
                content_status="",
                created="2024-06-09T21:17:00+00:00",
                identifier="",
                keywords="",
                language="",
                last_modified_by="Carlos Fernández Lorán",
                last_printed=None,
                modified="2024-06-09T21:27:00+00:00",
                revision=2,
                subject="",
                title="",
                version="",
            ),
        }

    def test_run_with_table(self, test_files_path, docx_converter):
        """
        Test if the component runs correctly
        """
        paths = [test_files_path / "docx" / "sample_docx.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "Donald Trump" in docs[0].content  ## :-)
        assert docs[0].meta.keys() == {"file_path", "docx"}
        assert docs[0].meta == {
            "file_path": str(paths[0]),
            "docx": DOCXMetadata(
                author="Saha, Anirban",
                category="",
                comments="",
                content_status="",
                created="2020-07-14T08:14:00+00:00",
                identifier="",
                keywords="",
                language="",
                last_modified_by="Saha, Anirban",
                last_printed=None,
                modified="2020-07-14T08:16:00+00:00",
                revision=1,
                subject="",
                title="",
                version="",
            ),
        }
        # let's now detect that the table markdown is correctly added and that order of elements is correct
        content_parts = docs[0].content.split("\n\n")
        table_index = next(i for i, part in enumerate(content_parts) if "| This | Is     | Just a |" in part)
        # check that natural order of the document is preserved
        assert any("Donald Trump" in part for part in content_parts[:table_index]), "Text before table not found"
        assert any(
            "Now we are in Page 2" in part for part in content_parts[table_index + 1 :]
        ), "Text after table not found"

    def test_run_with_additional_meta(self, test_files_path, docx_converter):
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths, meta={"language": "it", "author": "test_author"})
        doc = output["documents"][0]
        assert doc.meta == {
            "file_path": str(paths[0]),
            "docx": DOCXMetadata(
                author="Microsoft Office User",
                category="",
                comments="",
                content_status="",
                created="2024-06-09T21:17:00+00:00",
                identifier="",
                keywords="",
                language="",
                last_modified_by="Carlos Fernández Lorán",
                last_printed=None,
                modified="2024-06-09T21:27:00+00:00",
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

    def test_run_error_non_existent_file(self, docx_converter, caplog):
        """
        Test if the component correctly handles errors.
        """
        paths = ["non_existing_file.docx"]
        with caplog.at_level(logging.WARNING):
            docx_converter.run(sources=paths)
            assert "Could not read non_existing_file.docx" in caplog.text

    def test_run_page_breaks(self, test_files_path, docx_converter):
        """
        Test if the component correctly parses page breaks.
        """
        paths = [test_files_path / "docx" / "sample_docx_2_page_breaks.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content.count("\f") == 4

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

    def test_document_with_docx_metadata_to_dict(self):
        docx_metadata = DOCXMetadata(
            author="Microsoft Office User",
            category="category",
            comments="comments",
            content_status="",
            created="2024-06-09T21:17:00+00:00",
            identifier="",
            keywords="",
            language="",
            last_modified_by="Carlos Fernández Lorán",
            last_printed=None,
            modified="2024-06-09T21:27:00+00:00",
            revision=2,
            subject="",
            title="",
            version="",
        )
        doc = Document(content="content", meta={"test": 1, "docx": docx_metadata}, id="1")
        assert doc.to_dict(flatten=False) == {
            "blob": None,
            "dataframe": None,
            "content": "content",
            "id": "1",
            "score": None,
            "embedding": None,
            "sparse_embedding": None,
            "meta": {
                "test": 1,
                "docx": {
                    "author": "Microsoft Office User",
                    "category": "category",
                    "comments": "comments",
                    "content_status": "",
                    "created": "2024-06-09T21:17:00+00:00",
                    "identifier": "",
                    "keywords": "",
                    "language": "",
                    "last_modified_by": "Carlos Fernández Lorán",
                    "last_printed": None,
                    "modified": "2024-06-09T21:27:00+00:00",
                    "revision": 2,
                    "subject": "",
                    "title": "",
                    "version": "",
                },
            },
        }

        # check it is JSON serializable
        json_str = json.dumps(doc.to_dict(flatten=False))
        assert json.loads(json_str) == doc.to_dict(flatten=False)
