# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import PropertyMock, patch

import pytest
from oxmsg import Message

from haystack import Document
from haystack.components.converters.msg import MSGToDocument
from haystack.dataclasses import ByteStream


class TestMSGToDocument:
    def test_run_bytestream_without_file_path(self, test_files_path):
        """A bare ByteStream source (no 'file_path' in its meta) must not raise a KeyError."""
        converter = MSGToDocument(store_full_path=True)
        data = (test_files_path / "msg" / "sample.msg").read_bytes()
        bytestream = ByteStream(data=data)

        result = converter.run(sources=[bytestream], meta={"date_added": "2021-09-01T00:00:00"})

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta == {"date_added": "2021-09-01T00:00:00"}
        assert len(result["attachments"]) == 1
        assert result["attachments"][0].meta == {"date_added": "2021-09-01T00:00:00", "file_path": "sample_pdf_1.pdf"}
        assert "parent_file_path" not in result["attachments"][0].meta

    def test_run(self, test_files_path):
        converter = MSGToDocument(store_full_path=True)
        paths = [test_files_path / "msg" / "sample.msg"]
        result = converter.run(sources=paths, meta={"date_added": "2021-09-01T00:00:00"})
        assert len(result["documents"]) == 1
        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content is not None
        assert document.content.startswith('From: "Example Alice"')
        assert result["documents"][0].meta == {
            "date_added": "2021-09-01T00:00:00",
            "file_path": str(test_files_path / "msg" / "sample.msg"),
        }
        assert len(result["attachments"]) == 1
        attachment = result["attachments"][0]
        assert isinstance(attachment, ByteStream)
        assert attachment.mime_type == "application/pdf"
        assert result["attachments"][0].meta == {
            "date_added": "2021-09-01T00:00:00",
            "parent_file_path": str(test_files_path / "msg" / "sample.msg"),
            "file_path": "sample_pdf_1.pdf",
        }

    @pytest.mark.parametrize(
        ("headers", "expected_lines"),
        [
            (
                {"To": "custom to", "Cc": "custom cc", "Bcc": "custom bcc"},
                ["To: custom to", "Cc: custom cc", "Bcc: custom bcc"],
            ),
            (
                {},
                [
                    "To: Example Alice alice.tester@example.com",
                    "Cc: bob.recipient01@example.com bob.recipient01@example.com",
                    "Bcc: carol.receiver01@example.com carol.receiver01@example.com",
                ],
            ),
            (
                {"To": "", "cC": "custom cc", "Bcc": " \t"},
                [
                    "To: Example Alice alice.tester@example.com",
                    "Cc: custom cc",
                    "Bcc: carol.receiver01@example.com carol.receiver01@example.com",
                ],
            ),
        ],
        ids=["headers", "missing-headers", "strange-headers"],
    )
    def test_run_recipients_from_headers_or_table(self, test_files_path, headers, expected_lines):
        with patch.object(Message, "message_headers", new_callable=PropertyMock, return_value=headers):
            result = MSGToDocument().run(sources=[test_files_path / "msg" / "sample.msg"])

        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content is not None
        assert [
            line for line in document.content.splitlines() if line.startswith(("To:", "Cc:", "Bcc:"))
        ] == expected_lines

    def test_run_wrong_file_type(self, test_files_path, caplog):
        converter = MSGToDocument(store_full_path=False)
        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        result = converter.run(sources=paths, meta={"date_added": "2021-09-01T00:00:00"})
        assert len(result["documents"]) == 0
        assert "msg_file is not an Outlook MSG file" in caplog.text

    def test_run_empty_sources(self, test_files_path):
        converter = MSGToDocument(store_full_path=False)
        result = converter.run(sources=[])
        assert len(result["documents"]) == 0
        assert len(result["attachments"]) == 0
