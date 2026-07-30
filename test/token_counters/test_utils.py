# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.token_counters.utils import _render_message, _rendered_conversation, _tool_result_text

IMAGE = ImageContent(base64_image="Zm9v", mime_type="image/png")
FILE = FileContent(base64_data="Zm9v", mime_type="application/pdf", filename="report.pdf")


def _tool_result(result: str, *, error: bool = False) -> ChatMessage:
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name="search", arguments={}, id="c1"), error=error
    )


class TestRenderMessage:
    def test_renders_every_kind_of_message(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("what is haystack?"),
            ChatMessage.from_assistant(
                "let me look", tool_calls=[ToolCall(tool_name="search", arguments={"b": 2, "a": 1})]
            ),
            _tool_result("found it"),
            _tool_result("boom", error=True),
            ChatMessage.from_user(content_parts=["look:", IMAGE, FILE]),
            ChatMessage.from_assistant(),
        ]
        assert "\n".join(_render_message(message) for message in messages) == (
            "[system] rules\n"
            "[user] what is haystack?\n"
            "[assistant] let me look\n"
            '[assistant -> tool_call] search({"a": 1, "b": 2})\n'
            "[tool:search] found it\n"
            "[tool:search (error)] boom\n"
            "[user] look:\n"
            "[user] <image>\n"
            "[user] <file: report.pdf>\n"
            "[assistant] <no content>"
        )


class TestToolResultText:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            pytest.param("hello", "hello", id="plain-string"),
            pytest.param([TextContent(text="a"), TextContent(text="b")], "ab", id="text-blocks-concatenated"),
            pytest.param([TextContent(text="see "), IMAGE], "see <image>", id="non-text-placeholder"),
        ],
    )
    def test_tool_result_text(self, result, expected):
        assert _tool_result_text(result) == expected


class TestRenderedConversation:
    def test_joins_messages_with_newlines(self):
        messages = [ChatMessage.from_user("a"), ChatMessage.from_assistant("b")]

        assert _rendered_conversation(messages) == "[user] a\n[assistant] b"

    def test_empty_conversation(self):
        assert _rendered_conversation([]) == ""
