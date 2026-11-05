# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from haystack.components.converters.markdown import MarkdownToDocument
from haystack.dataclasses import ByteStream


class TestMarkdownToDocumentFrontmatter:
    @pytest.mark.parametrize("frontmatter", ["false", "0", "[]", '""', "true", "[item]"])
    def test_run_keeps_non_mapping_frontmatter_as_content_and_logs_warning(self, frontmatter, caplog):
        source = ByteStream.from_string(f"---\n{frontmatter}\n---\n# Body\nKeep this text.\n")
        expected = MarkdownToDocument(progress_bar=False).run(sources=[source])["documents"][0]
        converter = MarkdownToDocument(progress_bar=False, extract_frontmatter=True)

        with caplog.at_level(logging.WARNING):
            documents = converter.run(sources=[source])["documents"]

        assert len(documents) == 1
        assert documents[0].content == expected.content
        assert documents[0].meta == {}
        assert "expected a mapping" in caplog.text

    @pytest.mark.parametrize("frontmatter", ["", "null", "{}"])
    def test_run_removes_empty_frontmatter(self, frontmatter, caplog):
        source = ByteStream.from_string(f"---\n{frontmatter}\n---\n# Body\nKeep this text.\n")
        expected = MarkdownToDocument(progress_bar=False).run(
            sources=[ByteStream.from_string("# Body\nKeep this text.\n")]
        )["documents"][0]
        converter = MarkdownToDocument(progress_bar=False, extract_frontmatter=True)

        with caplog.at_level(logging.WARNING):
            documents = converter.run(sources=[source])["documents"]

        assert len(documents) == 1
        assert documents[0].content == expected.content
        assert documents[0].meta == {}
        assert not caplog.records
