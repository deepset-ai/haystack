# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.dataclasses.chat_message import ToolCallResultContentT
from haystack.hooks.tool_result_offloading import FileSystemToolResultStore, ToolResultStore
from haystack.hooks.tool_result_offloading.utils import _offloadable_content_blocks, _offloaded_message

# A 1x1 PNG and a minimal PDF.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n\xde\xad\xbe\xef\n%%EOF\n"
IMAGE_BLOCK = ImageContent(base64_image=base64.b64encode(PNG_BYTES).decode("utf-8"), mime_type="image/png")
FILE_BLOCK = FileContent(
    base64_data=base64.b64encode(PDF_BYTES).decode("utf-8"), mime_type="application/pdf", filename="report.pdf"
)


class TextOnlyToolResultStore(ToolResultStore):
    """A store that only holds text, leaving `supports_binary_content` at its False default."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def write(self, *, key: str, content: str | bytes) -> str:
        if isinstance(content, bytes):
            raise TypeError("Binary content not supported")
        self.data[key] = content
        return key

    def read(self, reference: str) -> str:
        return self.data[reference]


def _tool_message(result: ToolCallResultContentT, *, error: bool = False) -> ChatMessage:
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name="a", arguments={}, id="1"), error=error, meta={"source": "test"}
    )


class TestOffloadableContentBlocks:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param("", None, id="empty_string"),
            pytest.param([], None, id="empty_list"),
            pytest.param([TextContent(text="")], None, id="single_empty_text_block"),
            pytest.param([TextContent(text=""), TextContent(text="")], None, id="several_empty_text_blocks"),
            pytest.param("text", [TextContent(text="text")], id="string"),
            pytest.param(
                [TextContent(text=""), IMAGE_BLOCK], [TextContent(text=""), IMAGE_BLOCK], id="empty_text_with_image"
            ),
        ],
    )
    def test_offloadable_content_blocks(self, tmp_path, result, expected):
        tool_result = _tool_message(result).tool_call_result
        assert tool_result is not None

        assert _offloadable_content_blocks(result=tool_result, store=FileSystemToolResultStore(root=tmp_path)) == (
            expected
        )

    def test_text_only_store_rejects_binary_content(self, caplog):
        tool_result = _tool_message([TextContent("caption"), FILE_BLOCK]).tool_call_result
        assert tool_result is not None

        assert _offloadable_content_blocks(result=tool_result, store=TextOnlyToolResultStore()) is None
        assert "does not support binary content" in caplog.text


class TestOffloadedMessage:
    def test_text_result(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        offloaded = _offloaded_message(
            message=_tool_message("ABCDEFGH", error=True),
            content_blocks=[TextContent("ABCDEFGH")],
            store=store,
            key_prefix="key",
            preview_chars=5,
            additional_meta={"extra": 1},
        )

        reference = offloaded.meta["tool_result_offloaded"][0]
        assert Path(reference).name == "key.txt"
        assert store.read(reference) == "ABCDEFGH"
        assert offloaded.tool_call_result is not None
        assert offloaded.tool_call_result.result == (
            f"Tool result offloaded to text (8 characters) at '{reference}'. Preview: ABCDE..."
        )
        assert offloaded.tool_call_result.origin == ToolCall(tool_name="a", arguments={}, id="1")
        assert offloaded.tool_call_result.error is True
        assert offloaded.meta == {"source": "test", "extra": 1, "tool_result_offloaded": [reference]}

    def test_writes_every_block_to_its_own_entry(self, tmp_path):
        store = FileSystemToolResultStore(root=tmp_path)
        content: list[TextContent | ImageContent | FileContent] = [TextContent("caption"), IMAGE_BLOCK, FILE_BLOCK]
        offloaded = _offloaded_message(
            message=_tool_message(content), content_blocks=content, store=store, key_prefix="key", preview_chars=4
        )

        references = offloaded.meta["tool_result_offloaded"]
        assert [Path(reference).name for reference in references] == ["key_0.txt", "key_1.png", "key_2.pdf"]
        assert store.read(references[0]) == "caption"
        assert store.read(references[1]) == PNG_BYTES
        assert store.read(references[2]) == PDF_BYTES
        assert offloaded.tool_call_result is not None
        assert offloaded.tool_call_result.result == "\n".join(
            [
                "Tool result offloaded to 3 files:",
                f"1. text (7 characters) at '{references[0]}'. Preview: capt...",
                f"2. image/png ({len(PNG_BYTES)} bytes) at '{references[1]}'",
                f"3. application/pdf named 'report.pdf' ({len(PDF_BYTES)} bytes) at '{references[2]}'",
            ]
        )

    @pytest.mark.parametrize(
        ("block", "expected_extension"),
        [
            pytest.param(FileContent(base64_data="aGk=", mime_type="text/csv", filename=None), ".csv", id="mime_type"),
            pytest.param(
                FileContent(base64_data="aGk=", mime_type="application/pdf", filename="notes.md"), ".md", id="filename"
            ),
            pytest.param(FileContent(base64_data="aGk=", mime_type=None, filename=None), ".bin", id="fallback"),
        ],
    )
    def test_binary_block_extension(self, tmp_path, block, expected_extension):
        offloaded = _offloaded_message(
            message=_tool_message([block]),
            content_blocks=[block],
            store=FileSystemToolResultStore(root=tmp_path),
            key_prefix="key",
            preview_chars=0,
        )

        assert Path(offloaded.meta["tool_result_offloaded"][0]).suffix == expected_extension

    def test_rejects_a_message_without_a_tool_result(self, tmp_path):
        with pytest.raises(ValueError, match="Only tool-result messages can be offloaded."):
            _offloaded_message(
                message=ChatMessage.from_user("hi"),
                content_blocks=[TextContent("hi")],
                store=FileSystemToolResultStore(root=tmp_path),
                key_prefix="key",
                preview_chars=0,
            )
