# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, PropertyMock, patch

import pytest
from oxmsg import Message

from haystack import Document
from haystack.components.converters.msg import MSGToDocument
from haystack.dataclasses import ByteStream


class TestMSGToDocument:
    @pytest.mark.parametrize(
        "headers",
        [
            None,
            {},
            {"Cc": "custom cc"},
            {"BCC": "custom bcc"},
            {"CC": "cc", "Bcc": "bcc"},
            {"Cc": "", "Bcc": ""},
            {"CC": "", "BCC": ""},
            {"Cc": "", "CC": "custom cc", "Bcc": "", "BCC": "custom bcc"},
        ],
    )
    def test_run_recipient_lines(self, test_files_path, headers):
        source = test_files_path / "msg" / "sample.msg"
        original_headers = Message.load(source).message_headers
        resolved_headers = original_headers if headers is None else headers
        with patch.object(Message, "message_headers", new_callable=PropertyMock, return_value=resolved_headers):
            result = MSGToDocument().run(sources=[source])
        document = result["documents"][0]
        assert isinstance(document, Document)
        content = document.content
        assert content is not None
        cc = resolved_headers.get("Cc") or resolved_headers.get("CC")
        bcc = resolved_headers.get("Bcc") or resolved_headers.get("BCC")
        assert [line for line in content.splitlines() if line.startswith(("To:", "Cc:", "Bcc:"))] == [
            "To: Sebastian Lee sebastian.lee@deepset.ai",
            f"Cc: {cc if cc is not None else 'sebastian.lee+cc@deepset.ai sebastian.lee+cc@deepset.ai'}",
            f"Bcc: {bcc if bcc is not None else 'sebastian.lee+bcc@deepset.ai sebastian.lee+bcc@deepset.ai'}",
        ]

    @pytest.mark.parametrize("flag", [0, 0x10000000, 0x80000000, 0x90000000])
    def test_run_recipient_types_and_order(self, test_files_path, flag):
        recipients = []
        for index, kind in enumerate([3, 1, 2, 1, 0, None, 4, 17]):
            recip = Mock(email_address=f"user{index}@example.com")
            recip.name = f"User {index}"
            recip.properties.int_prop_value.return_value = None if kind is None else kind | flag
            recipients.append(recip)
        with (
            patch.object(Message, "recipients", new_callable=PropertyMock, return_value=recipients),
            patch.object(Message, "message_headers", new_callable=PropertyMock, return_value={}),
        ):
            result = MSGToDocument().run(sources=[test_files_path / "msg" / "sample.msg"])
        document = result["documents"][0]
        assert isinstance(document, Document)
        content = document.content
        assert content is not None
        assert [line for line in content.splitlines() if line.startswith(("To:", "Cc:", "Bcc:"))] == [
            "To: User 1 user1@example.com,User 3 user3@example.com",
            "Cc: User 2 user2@example.com",
            "Bcc: User 0 user0@example.com",
        ]

    def test_run_without_recipients(self, test_files_path):
        with (
            patch.object(Message, "recipients", new_callable=PropertyMock, return_value=()),
            patch.object(Message, "message_headers", new_callable=PropertyMock, return_value={}),
        ):
            result = MSGToDocument().run(sources=[test_files_path / "msg" / "sample.msg"])
        document = result["documents"][0]
        assert isinstance(document, Document)
        content = document.content
        assert content is not None
        assert not any(line.startswith(("To:", "Cc:", "Bcc:")) for line in content.splitlines())

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
        assert document.content.startswith('From: "Sebastian Lee"')
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
