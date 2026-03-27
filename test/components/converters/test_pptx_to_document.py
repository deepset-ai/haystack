# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest

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
        files = [str(test_files_path / "pptx" / "sample_pptx.pptx"), bytestream]
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
        assert docs[0].meta["file_path"] == os.path.basename(files[0])
        assert docs[1].meta == {"file_path": os.path.basename(bytestream.meta["file_path"]), "key": "value"}

    def test_run_error_non_existent_file(self, caplog):
        sources = ["non_existing_file.pptx"]
        converter = PPTXToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.pptx" in caplog.text
            assert results["documents"] == []

    def test_run_error_wrong_file_type(self, caplog, test_files_path):
        sources = [str(test_files_path / "txt" / "doc_1.txt")]
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
            PPTXToDocument(link_format="invalid")

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
