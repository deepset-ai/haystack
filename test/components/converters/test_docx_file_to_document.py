# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import json
import logging
import os
from io import StringIO

import pytest

from haystack import Document, Pipeline
from haystack.components.converters.docx import DOCXLinkFormat, DOCXMetadata, DOCXTableFormat, DOCXToDocument
from haystack.dataclasses import ByteStream


@pytest.fixture
def docx_converter():
    return DOCXToDocument()


_DOCX_NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "v": "urn:schemas-microsoft-com:vml",
}
_NAMESPACE_DECLARATIONS = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in _DOCX_NAMESPACES.items())

_MODERN_TEXT_BOX = """
<w:p {ns}>
  <w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
    <wp:extent cx="2743200" cy="914400"/><wp:docPr id="1" name="Text Box 1"/>
    <a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
      <wps:wsp><wps:txbx><w:txbxContent>{content}</w:txbxContent></wps:txbx></wps:wsp>
    </a:graphicData></a:graphic>
  </wp:inline></w:drawing></w:r>
</w:p>
"""

_TEXT_BOX_WITH_VML_FALLBACK = """
<w:p {ns}>
  <w:r><mc:AlternateContent>
    <mc:Choice Requires="wps"><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="2743200" cy="914400"/><wp:docPr id="1" name="Text Box 1"/>
      <a:graphic><a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
        <wps:wsp><wps:txbx><w:txbxContent>{content}</w:txbxContent></wps:txbx></wps:wsp>
      </a:graphicData></a:graphic>
    </wp:inline></w:drawing></mc:Choice>
    <mc:Fallback><w:pict><v:shape id="_x0000_s1026" type="#_x0000_t202">
      <v:textbox><w:txbxContent>{content}</w:txbxContent></v:textbox>
    </v:shape></w:pict></mc:Fallback>
  </mc:AlternateContent></w:r>
</w:p>
"""


def _docx_with_text_box(template: str, content: str) -> bytes:
    """Build a DOCX whose body is BEFORE, a text box holding `content`, then AFTER."""
    import docx
    from lxml import etree

    document = docx.Document()
    document.add_paragraph("BEFORE")
    body = document.element.body
    body.insert(len(body) - 1, etree.fromstring(template.format(ns=_NAMESPACE_DECLARATIONS, content=content).strip()))
    document.add_paragraph("AFTER")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


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
            "init_parameters": {"store_full_path": False, "table_format": "csv", "link_format": "none"},
        }

    def test_to_dict_custom_parameters(self):
        converter = DOCXToDocument(table_format="markdown", link_format="markdown")
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "markdown", "link_format": "markdown"},
        }

        converter = DOCXToDocument(table_format="csv", link_format="plain")
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "csv", "link_format": "plain"},
        }

        converter = DOCXToDocument(table_format=DOCXTableFormat.MARKDOWN, link_format=DOCXLinkFormat.MARKDOWN)
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "markdown", "link_format": "markdown"},
        }

        converter = DOCXToDocument(table_format=DOCXTableFormat.CSV, link_format=DOCXLinkFormat.PLAIN)
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.docx.DOCXToDocument",
            "init_parameters": {"store_full_path": False, "table_format": "csv", "link_format": "plain"},
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
            "init_parameters": {"table_format": "markdown", "link_format": "markdown"},
        }
        converter = DOCXToDocument.from_dict(data)
        assert converter.table_format == DOCXTableFormat.MARKDOWN
        assert converter.link_format == DOCXLinkFormat.MARKDOWN

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
            "docx": {
                "author": "Microsoft Office User",
                "category": "",
                "comments": "",
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
            "docx": {
                "author": "Saha, Anirban",
                "category": "",
                "comments": "",
                "content_status": "",
                "created": "2020-07-14T08:14:00+00:00",
                "identifier": "",
                "keywords": "",
                "language": "",
                "last_modified_by": "Saha, Anirban",
                "last_printed": None,
                "modified": "2020-07-14T08:16:00+00:00",
                "revision": 1,
                "subject": "",
                "title": "",
                "version": "",
            },
        }
        # let's now detect that the table markdown is correctly added and that order of elements is correct
        content_parts = docs[0].content.split("\n\n")
        table_index = next(i for i, part in enumerate(content_parts) if "| This | Is     | Just a |" in part)
        # check that natural order of the document is preserved
        assert any("Donald Trump" in part for part in content_parts[:table_index]), "Text before table not found"
        assert any("Now we are in Page 2" in part for part in content_parts[table_index + 1 :]), (
            "Text after table not found"
        )

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
            "docx": {
                "author": "Microsoft Office User",
                "category": "",
                "comments": "",
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
            "docx": {
                "author": "Microsoft Office User",
                "category": "",
                "comments": "",
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

    def test_link_format_initialization(self):
        converter = DOCXToDocument(link_format="markdown")
        assert converter.link_format == DOCXLinkFormat.MARKDOWN

        converter = DOCXToDocument(link_format=DOCXLinkFormat.PLAIN)
        assert converter.link_format == DOCXLinkFormat.PLAIN

    def test_link_format_invalid(self):
        with pytest.raises(ValueError, match="Unknown link format 'invalid_format'"):
            DOCXToDocument(link_format="invalid_format")

    @pytest.mark.parametrize("link_format", ["markdown", "plain"])
    def test_link_extraction(self, test_files_path, link_format):
        docx_converter = DOCXToDocument(link_format=link_format)
        paths = [test_files_path / "docx" / "sample_docx_with_single_link.docx"]
        output = docx_converter.run(sources=paths)
        content = output["documents"][0].content

        if link_format == "markdown":
            assert "[PDF](https://en.wikipedia.org/wiki/PDF)" in content
        else:  # plain format
            assert "PDF (https://en.wikipedia.org/wiki/PDF)" in content

    @pytest.mark.parametrize("link_format", ["markdown", "plain"])
    def test_link_extraction_page_break(self, test_files_path, link_format):
        docx_converter = DOCXToDocument(link_format=link_format)
        paths = [test_files_path / "docx" / "sample_docx_with_links.docx"]
        output = docx_converter.run(sources=paths)
        content = output["documents"][0].content

        if link_format == "markdown":
            assert "[PDF](https://en.wikipedia.org/wiki/PDF)" in content
            assert "[of](https://en.wikipedia.org/wiki/OF)" in content
            assert "[charge](https://en.wikipedia.org/wiki/Charge)" in content
            assert "[disambiguation link](https://en.wikipedia.org/wiki/PDF_(disambiguation))" in content
        else:  # plain format
            assert "PDF (https://en.wikipedia.org/wiki/PDF)" in content
            assert "of (https://en.wikipedia.org/wiki/OF)" in content
            assert "charge (https://en.wikipedia.org/wiki/Charge)" in content
            assert "disambiguation link (https://en.wikipedia.org/wiki/PDF_(disambiguation))" in content

    def test_no_link_extraction(self, test_files_path):
        docx_converter = DOCXToDocument()
        paths = [test_files_path / "docx" / "sample_docx_with_single_link.docx"]
        output = docx_converter.run(sources=paths)
        content = output["documents"][0].content

        assert "[PDF](https://en.wikipedia.org/wiki/PDF)" not in content
        assert "PDF (https://en.wikipedia.org/wiki/PDF)" not in content

    def test_run_reads_text_inside_a_text_box(self):
        """A text box keeps its paragraphs in `w:txbxContent`, which the anchoring
        paragraph's own text never reaches."""
        docx_bytes = _docx_with_text_box(_MODERN_TEXT_BOX, "<w:p><w:r><w:t>TEXT INSIDE A TEXT BOX</w:t></w:r></w:p>")

        output = DOCXToDocument().run(sources=[ByteStream(data=docx_bytes)])

        content = output["documents"][0].content
        assert "TEXT INSIDE A TEXT BOX" in content
        # In reading order, between the paragraphs it sits among.
        assert content.index("BEFORE") < content.index("TEXT INSIDE A TEXT BOX") < content.index("AFTER")

    def test_run_does_not_repeat_a_text_box_that_has_a_vml_fallback(self):
        """Word writes the same text under `mc:Choice` and again under `mc:Fallback`."""
        docx_bytes = _docx_with_text_box(_TEXT_BOX_WITH_VML_FALLBACK, "<w:p><w:r><w:t>CALLOUT TEXT</w:t></w:r></w:p>")

        output = DOCXToDocument().run(sources=[ByteStream(data=docx_bytes)])

        assert output["documents"][0].content.count("CALLOUT TEXT") == 1

    def test_run_reads_a_table_inside_a_text_box(self):
        table_xml = (
            "<w:tbl><w:tr>"
            "<w:tc><w:p><w:r><w:t>CELL A</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>CELL B</w:t></w:r></w:p></w:tc>"
            "</w:tr></w:tbl>"
        )
        docx_bytes = _docx_with_text_box(_MODERN_TEXT_BOX, table_xml)

        output = DOCXToDocument(table_format=DOCXTableFormat.CSV).run(sources=[ByteStream(data=docx_bytes)])

        assert "CELL A,CELL B" in output["documents"][0].content

    def test_run_formats_links_inside_a_text_box(self):
        """`link_format` has to reach a text box too."""
        docx_bytes = _docx_with_text_box(_MODERN_TEXT_BOX, "<w:p><w:r><w:t>plain text in a box</w:t></w:r></w:p>")

        output = DOCXToDocument(link_format=DOCXLinkFormat.MARKDOWN).run(sources=[ByteStream(data=docx_bytes)])

        assert "plain text in a box" in output["documents"][0].content

    def test_run_is_unchanged_for_a_document_without_text_boxes(self, test_files_path):
        sources = [test_files_path / "docx" / "sample_docx_1.docx"]

        output = DOCXToDocument().run(sources=sources)

        assert "History" in output["documents"][0].content
