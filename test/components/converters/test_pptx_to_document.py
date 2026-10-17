# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import os
from pathlib import Path

import pytest

from haystack.components.converters.pptx import PPTXToDocument
from haystack.dataclasses import ByteStream


def _deck_with_a_group_and_a_table() -> bytes:
    """A one-slide deck holding a plain textbox, a table, and a group of textboxes."""
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])  # blank

    textbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(4), Inches(0.6))
    textbox.text_frame.text = "PLAIN TEXTBOX"

    table = slide.shapes.add_table(2, 2, Inches(0.5), Inches(1.2), Inches(4), Inches(1.2)).table
    table.cell(0, 0).text = "HEADER A"
    table.cell(0, 1).text = "HEADER B"
    table.cell(1, 0).text = "CELL A"
    table.cell(1, 1).text = "CELL B"

    first = slide.shapes.add_textbox(Inches(0.5), Inches(3.0), Inches(2), Inches(0.5))
    first.text_frame.text = "GROUPED ONE"
    second = slide.shapes.add_textbox(Inches(3.0), Inches(3.0), Inches(2), Inches(0.5))
    second.text_frame.text = "GROUPED TWO"
    slide.shapes.add_group_shape([first, second])

    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


class TestPPTXToDocument:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        bytestream = ByteStream.from_file_path(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["file_path"] = str(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["key"] = "value"
        first_path = str(test_files_path / "pptx" / "sample_pptx.pptx")
        files: list[str | Path | ByteStream] = [first_path, bytestream]
        converter = PPTXToDocument()
        output = converter.run(sources=files)
        docs = output["documents"]

        assert len(docs) == 2
        assert (
            "Sample Title Slide\nJane Doe\fTitle of First Slide\nThis is a bullet point\nThis is another bullet point"
            in docs[0].content
        )
        assert (
            "Sample Title Slide\nJane Doe\fTitle of First Slide\nThis is a bullet point\nThis is another bullet point"
            in docs[0].content
        )
        assert docs[0].meta["file_path"] == os.path.basename(first_path)
        assert docs[1].meta == {"file_path": os.path.basename(bytestream.meta["file_path"]), "key": "value"}

    def test_run_error_non_existent_file(self, caplog):
        sources: list[str | Path | ByteStream] = ["non_existing_file.pptx"]
        converter = PPTXToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.pptx" in caplog.text
            assert results["documents"] == []

    def test_run_error_wrong_file_type(self, caplog, test_files_path):
        sources: list[str | Path | ByteStream] = [str(test_files_path / "txt" / "doc_1.txt")]
        converter = PPTXToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "doc_1.txt and convert it" in caplog.text
            assert results["documents"] == []

    def test_run_with_meta(self, test_files_path):
        bytestream = ByteStream.from_file_path(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["file_path"] = str(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["key"] = "value"

        converter = PPTXToDocument()
        output = converter.run(sources=[bytestream], meta=[{"language": "it"}])
        document = output["documents"][0]

        assert document.meta == {
            "file_path": os.path.basename(test_files_path / "pptx" / "sample_pptx.pptx"),
            "key": "value",
            "language": "it",
        }

    def test_run_with_store_full_path_false(self, test_files_path):
        """
        Test if the component runs correctly with store_full_path=False
        """
        bytestream = ByteStream.from_file_path(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["file_path"] = str(test_files_path / "pptx" / "sample_pptx.pptx")
        bytestream.meta["key"] = "value"

        converter = PPTXToDocument(store_full_path=False)
        output = converter.run(sources=[bytestream], meta=[{"language": "it"}])
        document = output["documents"][0]

        assert document.meta == {"file_path": "sample_pptx.pptx", "key": "value", "language": "it"}

    def test_to_dict(self):
        converter = PPTXToDocument(link_format="markdown", store_full_path=True)
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.pptx.PPTXToDocument",
            "init_parameters": {"link_format": "markdown", "store_full_path": True},
        }

    def test_to_dict_defaults(self):
        converter = PPTXToDocument()
        data = converter.to_dict()
        assert data == {
            "type": "haystack.components.converters.pptx.PPTXToDocument",
            "init_parameters": {"link_format": "none", "store_full_path": False},
        }

    def test_link_format_invalid(self):
        with pytest.raises(ValueError, match="Unknown link format"):
            PPTXToDocument(link_format="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("link_format", ["markdown", "plain"])
    def test_link_extraction(self, test_files_path, link_format):
        converter = PPTXToDocument(link_format=link_format)
        paths = [test_files_path / "pptx" / "sample_pptx_with_link.pptx"]
        output = converter.run(sources=paths)
        content = output["documents"][0].content

        if link_format == "markdown":
            assert "[Example](https://example.com)" in content
        else:
            assert "Example (https://example.com)" in content

    def test_no_link_extraction(self, test_files_path):
        converter = PPTXToDocument()
        paths = [test_files_path / "pptx" / "sample_pptx_with_link.pptx"]
        output = converter.run(sources=paths)
        content = output["documents"][0].content

        assert "https://example.com" not in content
        assert "Example" in content

    def test_run_reads_text_inside_a_group(self):
        """A group carries no text of its own, so its children have to be read through."""
        converter = PPTXToDocument()

        output = converter.run(sources=[ByteStream(data=_deck_with_a_group_and_a_table())])

        content = output["documents"][0].content
        assert "GROUPED ONE" in content
        assert "GROUPED TWO" in content

    def test_run_reads_table_cells(self):
        """A table lives on a graphic frame, which is not a text frame either."""
        converter = PPTXToDocument()

        output = converter.run(sources=[ByteStream(data=_deck_with_a_group_and_a_table())])

        content = output["documents"][0].content
        assert "HEADER A,HEADER B" in content
        assert "CELL A,CELL B" in content

    def test_run_still_reads_a_plain_textbox(self):
        converter = PPTXToDocument()

        output = converter.run(sources=[ByteStream(data=_deck_with_a_group_and_a_table())])

        assert "PLAIN TEXTBOX" in output["documents"][0].content

    def test_run_formats_links_inside_a_group(self):
        """`link_format` has to survive the descent into a group."""
        from pptx import Presentation
        from pptx.util import Inches

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        textbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(4), Inches(0.6))
        run = textbox.text_frame.paragraphs[0].add_run()
        run.text = "deepset"
        run.hyperlink.address = "https://deepset.ai"
        slide.shapes.add_group_shape([textbox])
        buffer = io.BytesIO()
        presentation.save(buffer)

        converter = PPTXToDocument(link_format="markdown")
        output = converter.run(sources=[ByteStream(data=buffer.getvalue())])

        assert "[deepset](https://deepset.ai)" in output["documents"][0].content
