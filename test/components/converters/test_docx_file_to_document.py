import json
import os
import logging
import pytest
import csv
from io import StringIO

from haystack import Document, Pipeline
from haystack.components.converters.docx import DOCXMetadata, DOCXToDocument, DOCXTableFormat
from haystack.dataclasses import ByteStream


@pytest.fixture
def docx_converter():
    return DOCXToDocument()


class TestDOCXToDocument:
    def test_init(self, docx_converter):
        assert isinstance(docx_converter, DOCXToDocument)

    def test_init_with_string(self):
        converter = DOCXToDocument(table_format="markdown")
        assert isinstance(converter, DOCXToDocument)
        assert converter.table_format == DOCXTableFormat.MARKDOWN

    def test_init_with_invalid_string(self):
        with pytest.raises(ValueError, match="Unknown table format 'invalid_format'"):
            DOCXToDocument(table_format="invalid_format")

    def test_to_dict(self):
        converter = DOCXToDocument()
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "csv"},
        }

    def test_to_dict_custom_parameters(self):
        converter = DOCXToDocument(table_format="markdown")
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "markdown"},
        }

        converter = DOCXToDocument(table_format="csv")
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "csv"},
        }

        converter = DOCXToDocument(table_format=DOCXTableFormat.MARKDOWN)
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "markdown"},
        }

        converter = DOCXToDocument(table_format=DOCXTableFormat.CSV)
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "csv"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"table_format": "csv"},
        }
        converter = DOCXToDocument.from_dict(data)
        assert converter.table_format == DOCXTableFormat.CSV

    def test_from_dict_custom_parameters(self):
        data = {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"table_format": "markdown"},
        }
        converter = DOCXToDocument.from_dict(data)
        assert converter.table_format == DOCXTableFormat.MARKDOWN

    def test_from_dict_invalid_table_format(self):
        data = {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"table_format": "invalid_format"},
        }
        with pytest.raises(ValueError, match="Unknown table format 'invalid_format'"):
            DOCXToDocument.from_dict(data)

    def test_from_dict_empty_init_parameters(self):
        data = {"type": "haystack.components.converters.docx.DOCXToDocument", "init_parameters": {}}
        converter = DOCXToDocument.from_dict(data)
        assert converter.table_format == DOCXTableFormat.CSV

    def test_pipeline_serde(self):
        pipeline = Pipeline()
        converter = DOCXToDocument(table_format=DOCXTableFormat.MARKDOWN)
        pipeline.add_component("converter", converter)

        pipeline_str = pipeline.dumps()
        assert "haystack.components.converters.docx.DOCXToDocument" in pipeline_str
        assert "table_format" in pipeline_str
        assert "markdown" in pipeline_str

        new_pipeline = Pipeline.loads(pipeline_str)
        new_converter = new_pipeline.get_component("converter")
        assert isinstance(new_converter, DOCXToDocument)
        assert new_converter.table_format == DOCXTableFormat.MARKDOWN

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
            "file_path": os.path.basename(paths[0]),
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

    def test_run_with_table(self, test_files_path):
        """
        Test if the component runs correctly
        """
        docx_converter = DOCXToDocument(table_format=DOCXTableFormat.MARKDOWN)
        paths = [test_files_path / "docx" / "sample_docx.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "Donald Trump" in docs[0].content  ## :-)
        assert docs[0].meta.keys() == {"file_path", "docx"}
        assert docs[0].meta == {
            "file_path": os.path.basename(paths[0]),
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

    def test_run_with_store_full_path_false(self, test_files_path):
        """
        Test if the component runs correctly with store_full_path=False
        """
        docx_converter = DOCXToDocument(store_full_path=False)
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths)
        docs = output["documents"]
        assert len(docs) == 1
        assert "History" in docs[0].content
        assert docs[0].meta.keys() == {"file_path", "docx"}
        assert docs[0].meta == {
            "file_path": "sample_docx_1.docx",
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

    @pytest.mark.parametrize("table_format", ["markdown", "csv"])
    def test_table_between_two_paragraphs(self, test_files_path, table_format):
        docx_converter = DOCXToDocument(table_format=table_format)
        paths = [test_files_path / "docx" / "sample_docx_3.docx"]
        output = docx_converter.run(sources=paths)

        content = output["documents"][0].content

        paragraphs_one = content.find("Table: AI Use Cases in Different Industries")
        paragraphs_two = content.find("Paragraph 2:")
        table = content[
            paragraphs_one + len("Table: AI Use Cases in Different Industries") + 1 : paragraphs_two
        ].strip()

        if table_format == "markdown":
            split = list(filter(None, table.split("\n")))
            expected_table_header = "| Industry   | AI Use Case                    | Impact                    |"
            expected_last_row = "| Finance    | Fraud detection and prevention | Reduced financial losses  |"

            assert split[0] == expected_table_header
            assert split[-1] == expected_last_row
        if table_format == "csv":  # CSV format
            csv_reader = csv.reader(StringIO(table))
            rows = list(csv_reader)
            assert len(rows) == 3  # Header + 2 data rows
            assert rows[0] == ["Industry", "AI Use Case", "Impact"]
            assert rows[-1] == ["Finance", "Fraud detection and prevention", "Reduced financial losses"]

    @pytest.mark.parametrize("table_format", ["markdown", "csv"])
    def test_table_content_correct_parsing(self, test_files_path, table_format):
        docx_converter = DOCXToDocument(table_format=table_format)
        paths = [test_files_path / "docx" / "sample_docx_3.docx"]
        output = docx_converter.run(sources=paths)
        content = output["documents"][0].content

        paragraphs_one = content.find("Table: AI Use Cases in Different Industries")
        paragraphs_two = content.find("Paragraph 2:")
        table = content[
            paragraphs_one + len("Table: AI Use Cases in Different Industries") + 1 : paragraphs_two
        ].strip()

        if table_format == "markdown":
            split = list(filter(None, table.split("\n")))
            assert len(split) == 4

            expected_table_header = "| Industry   | AI Use Case                    | Impact                    |"
            expected_table_top_border = "| ---------- | ------------------------------ | ------------------------- |"
            expected_table_row_one = "| Healthcare | Predictive diagnostics         | Improved patient outcomes |"
            expected_table_row_two = "| Finance    | Fraud detection and prevention | Reduced financial losses  |"

            assert split[0] == expected_table_header
            assert split[1] == expected_table_top_border
            assert split[2] == expected_table_row_one
            assert split[3] == expected_table_row_two
        if table_format == "csv":  # CSV format
            csv_reader = csv.reader(StringIO(table))
            rows = list(csv_reader)
            assert len(rows) == 3  # Header + 2 data rows

            expected_header = ["Industry", "AI Use Case", "Impact"]
            expected_row_one = ["Healthcare", "Predictive diagnostics", "Improved patient outcomes"]
            expected_row_two = ["Finance", "Fraud detection and prevention", "Reduced financial losses"]

            assert rows[0] == expected_header
            assert rows[1] == expected_row_one
            assert rows[2] == expected_row_two

    def test_run_with_additional_meta(self, test_files_path, docx_converter):
        paths = [test_files_path / "docx" / "sample_docx_1.docx"]
        output = docx_converter.run(sources=paths, meta={"language": "it", "author": "test_author"})
        doc = output["documents"][0]
        assert doc.meta == {
            "file_path": os.path.basename(paths[0]),
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
