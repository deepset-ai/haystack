# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, ReasoningContent, TextContent, ToolCall
from haystack.token_counters.utils import (
    _non_text_placeholder,
    _render_message,
    _rendered_conversation,
    _rendered_tools,
    _tool_result_text,
)
from haystack.tools import tool

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


@tool
def search(query: Annotated[str, "the search query"]) -> str:
    """Search the web for a query."""
    return "result"


class TestNonTextPlaceholder:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            pytest.param(IMAGE, "<image>", id="image"),
            pytest.param(FILE, "<file: report.pdf>", id="file-with-name"),
            pytest.param(
                FileContent(base64_data="Zm9v", mime_type="application/pdf"), "<file: unnamed>", id="file-unnamed"
            ),
            # Anything else falls back to naming its type, so an unexpected block still shows up in the text.
            pytest.param(ReasoningContent(reasoning_text="thinking"), "<ReasoningContent>", id="unknown-type"),
        ],
    )
    def test_placeholder(self, content, expected):
        assert _non_text_placeholder(content) == expected


class TestRenderedTools:
    def test_no_tools_renders_nothing(self):
        assert _rendered_tools(None) == ""
        assert _rendered_tools([]) == ""

    def test_renders_the_schema_a_provider_would_be_sent(self):
        rendered = _rendered_tools([search])

        assert "search" in rendered
        assert "Search the web for a query." in rendered
        assert "the search query" in rendered
