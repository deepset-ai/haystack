# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ImageContent, TextContent, ToolCall
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _conversation_chars,
    _preserved_prefix_end,
    _safe_cut_index,
    _tool_result_text,
)

IMAGE = ImageContent(base64_image="Zm9v", mime_type="image/png")
COMPACTED_META = {_COMPACTION_META_KEY: {"strategy": "sliding_window"}}


def _tool_result(name: str, result: str, *, error: bool = False, call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_tool(
        tool_result=result, origin=ToolCall(tool_name=name, arguments={}, id=call_id), error=error
    )


def _tool_call(name: str, *, call_id: str = "c1", **arguments) -> ChatMessage:
    return ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name=name, arguments=arguments, id=call_id)])


class TestPreservedPrefixEnd:
    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param([], 0, id="empty"),
            pytest.param([ChatMessage.from_user("hi")], 0, id="no-leading-system"),
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], 2, id="only-system"),
            pytest.param(
                [ChatMessage.from_system("a"), ChatMessage.from_system("b"), ChatMessage.from_user("hi")],
                2,
                id="leading-system-block",
            ),
            pytest.param(
                [ChatMessage.from_user("hi"), ChatMessage.from_system("late rules")], 0, id="system-not-leading"
            ),
            # A previous summary sits right after the real system prompt. Counting it as prefix would leave it behind on
            # every future compaction instead of folding it into the next one.
            pytest.param(
                [
                    ChatMessage.from_system("rules"),
                    ChatMessage.from_system("summary", meta=COMPACTED_META),
                    ChatMessage.from_user("hi"),
                ],
                1,
                id="stops-at-previous-summary",
            ),
        ],
    )
    def test_prefix_end(self, messages, expected):
        assert _preserved_prefix_end(messages) == expected


class TestSafeCutIndex:
    @pytest.mark.parametrize(
        ("messages", "prefix_end", "keep_last_n", "expected"),
        [
            pytest.param([], 0, 5, 0, id="empty"),
            pytest.param([ChatMessage.from_user(f"m{i}") for i in range(6)], 0, 2, 4, id="keeps-requested-count"),
            # Cutting at index 2 would start the tail with a tool result whose call had been removed, so the boundary
            # walks back onto the assistant message holding that call.
            pytest.param(
                [ChatMessage.from_user("hi"), _tool_call("search"), _tool_result("search", "found")],
                0,
                1,
                1,
                id="walks-back-off-tool-result",
            ),
            pytest.param(
                [
                    ChatMessage.from_user("hi"),
                    ChatMessage.from_assistant(
                        tool_calls=[ToolCall(tool_name="search", arguments={}, id=f"c{i}") for i in range(3)]
                    ),
                    _tool_result("search", "a", call_id="c0"),
                    _tool_result("search", "b", call_id="c1"),
                    _tool_result("search", "c", call_id="c2"),
                ],
                0,
                1,
                1,
                id="walks-back-over-parallel-batch",
            ),
            pytest.param(
                [ChatMessage.from_system("rules"), _tool_result("search", "orphan")],
                1,
                1,
                1,
                id="never-moves-before-prefix",
            ),
            pytest.param(
                [ChatMessage.from_system("rules"), ChatMessage.from_user("hi")], 1, 50, 1, id="clamps-to-prefix"
            ),
        ],
    )
    def test_cut_index(self, messages, prefix_end, keep_last_n, expected):
        assert _safe_cut_index(messages, prefix_end=prefix_end, keep_last_n=keep_last_n) == expected


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


class TestConversationChars:
    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param([], 0, id="empty"),
            pytest.param([ChatMessage.from_user("12345")], 5, id="message-text"),
            pytest.param(
                [_tool_call("search", query="haystack")],
                len("search") + len('{"query": "haystack"}'),
                id="tool-call-name-and-arguments",
            ),
            pytest.param([_tool_result("search", "R" * 40)], 40, id="tool-result-content"),
        ],
    )
    def test_size(self, messages, expected):
        assert _conversation_chars(messages) == expected

    def test_shrinks_when_a_result_is_rewritten_in_place(self):
        # A strategy that replaces a result's content without removing the message must still register as a shrink,
        # which is why size is measured in characters rather than in messages.
        before = [_tool_call("search"), _tool_result("search", "R" * 500)]
        after = [_tool_call("search"), _tool_result("search", "[removed]")]
        assert len(after) == len(before)
        assert _conversation_chars(after) < _conversation_chars(before)
