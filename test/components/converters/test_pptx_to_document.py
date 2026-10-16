# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import os
from pathlib import Path
from typing import Literal

import pytest
from pptx import Presentation
from pptx.util import Inches

from haystack.components.converters.pptx import PPTXToDocument
from haystack.dataclasses import ByteStream


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

    @staticmethod
    def _pptx_with_soft_line_break(text: str) -> bytes:
        """Build a one-slide deck whose text box holds `text`, with \\v marking soft line breaks."""
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        text_frame = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(7), Inches(2)).text_frame
        text_frame.text = text
        buffer = io.BytesIO()
        presentation.save(buffer)
        return buffer.getvalue()

    @pytest.mark.parametrize("link_format", ["markdown", "plain"])
    def test_soft_line_break_with_link_format(self, link_format: Literal["markdown", "plain"]) -> None:
        """
        A soft line break (<a:br/>) between runs must survive every link format.

        See https://github.com/deepset-ai/haystack/issues/13043.
        """
        data = self._pptx_with_soft_line_break("Account: Alice\vBalance: 100")
        converter = PPTXToDocument(link_format=link_format)
        content = converter.run(sources=[ByteStream(data=data)])["documents"][0].content

        assert content == "Account: Alice\vBalance: 100"

    @pytest.mark.parametrize("link_format", ["markdown", "plain"])
    def test_soft_line_break_matches_link_format_none(self, link_format: Literal["markdown", "plain"]) -> None:
        """
        The formatted link modes return the same text the default "none" mode already returns.
        """
        data = self._pptx_with_soft_line_break("before\vafter")
        expected = PPTXToDocument(link_format="none").run(sources=[ByteStream(data=data)])["documents"][0].content
        assert (
            PPTXToDocument(link_format=link_format).run(sources=[ByteStream(data=data)])["documents"][0].content
            == expected
        )

    @pytest.mark.parametrize(
        ("link_format", "expected"),
        [("markdown", "before\v[example](https://example.com)"), ("plain", "before\vexample (https://example.com)")],
    )
    def test_hyperlink_still_formatted_next_to_soft_line_break(
        self, link_format: Literal["markdown", "plain"], expected: str
    ) -> None:
        """
        Keeping soft line breaks must not change how hyperlinked runs are formatted.
        """
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        text_frame = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(7), Inches(2)).text_frame
        text_frame.text = "before\vexample"
        text_frame.paragraphs[0].runs[1].hyperlink.address = "https://example.com"
        buffer = io.BytesIO()
        presentation.save(buffer)

        content = (
            PPTXToDocument(link_format=link_format)
            .run(sources=[ByteStream(data=buffer.getvalue())])["documents"][0]
            .content
        )
        assert content == expected
