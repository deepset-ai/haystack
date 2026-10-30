# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import pytest

from haystack.components.converters.subtitle import SubtitleToDocument, _parse_cues
from haystack.dataclasses import ByteStream


class TestSubtitleToDocument:
    def test_parse_srt_cues(self) -> None:
        text = (
            "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n"
            "2\n00:00:05,000 --> 00:00:06,500\nWorld\n"
        )
        cues = _parse_cues(text)
        assert cues == [(1.0, 4.0, "Hello"), (5.0, 6.5, "World")]

    def test_parse_vtt_strips_note_and_tags(self) -> None:
        text = (
            "WEBVTT\n\nNOTE ignore me\n\n"
            "00:00:01.000 --> 00:00:02.000\n<i>Hi</i>\n"
        )
        cues = _parse_cues(text)
        assert cues == [(1.0, 2.0, "Hi")]

    def test_run_srt_default_grouping(self, test_files_path: Path) -> None:
        path = test_files_path / "subtitles" / "sample.srt"
        converter = SubtitleToDocument(cues_per_document=2)
        docs = converter.run(sources=[path])["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Hello from SRT\nSecond cue here"
        assert docs[0].meta["start_time"] == 1.0
        assert docs[0].meta["end_time"] == 7.0
        assert docs[1].content == "Third cue"
        assert docs[1].meta["start_time"] == 8.0
        assert docs[1].meta["end_time"] == 10.0
        assert docs[0].meta["file_path"] == "sample.srt"

    def test_run_one_document_per_file(self, test_files_path: Path) -> None:
        path = test_files_path / "subtitles" / "sample.vtt"
        converter = SubtitleToDocument(cues_per_document=None)
        docs = converter.run(sources=[path])["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Hello from VTT\nSecond VTT cue"
        assert docs[0].meta["start_time"] == 1.0
        assert docs[0].meta["end_time"] == 6.0

    def test_run_no_cues_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        converter = SubtitleToDocument()
        stream = ByteStream(data=b"WEBVTT\n\nNOTE only\n", meta={"file_path": "empty.vtt"})
        with caplog.at_level(logging.WARNING):
            docs = converter.run(sources=[stream])["documents"]
            assert "No valid subtitle cues" in caplog.text
        assert docs == []

    def test_invalid_cues_per_document(self) -> None:
        with pytest.raises(ValueError):
            SubtitleToDocument(cues_per_document=0)
