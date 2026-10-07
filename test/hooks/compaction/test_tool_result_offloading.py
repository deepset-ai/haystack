# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from pathlib import Path

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.dataclasses.chat_message import ChatMessageContentT
from haystack.hooks.compaction import CompactionHook, ToolResultOffloadCompactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.hooks.tool_result_offloading import FileSystemToolResultStore
from haystack.hooks.tool_result_offloading.utils import _content_block_payload
from haystack.token_counters.utils import _rendered_conversation
from haystack.tools import ToolsType
from test.hooks.compaction.helpers import FakeCounter, conversation, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COUNTER = FakeCounter(chars_per_token=1)

# A 1x1 PNG, padded so its payload clearly outweighs the reference replacing it.
PNG_BYTES = (
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    + b"\x00" * 1000
)


def _payload_placeholder(content: ChatMessageContentT) -> str:
    """Render an image or a file as its base64 payload."""
    assert isinstance(content, (TextContent, ImageContent, FileContent))
    return _content_block_payload(content)


class PayloadCounter(FakeCounter):
    """A counter that measures an image or a file by its base64 payload, the way a provider-side counter would."""

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        return len(_rendered_conversation(messages, placeholder=_payload_placeholder)) // self.chars_per_token


def _step(content: list[TextContent | ImageContent | FileContent], call_id: str) -> list[ChatMessage]:
    """An Agent step whose single tool result carries the given content blocks."""
    call = tool_call(call_id)
    return [call, ChatMessage.from_tool(tool_result=content, origin=call.tool_calls[0])]


INELIGIBLE_CONVERSATION = [
    ChatMessage.from_user("task"),
    tool_call("small"),
    tool_result("small", call_id="small"),
    tool_call("error"),
    tool_result("error" * 100, call_id="error", error=True),
    tool_call("offloaded"),
    ChatMessage.from_tool(
        tool_result="offloaded" * 100,
        origin=ToolCall(tool_name="search", arguments={}, id="offloaded"),
        meta={"tool_result_offloaded": ["stored-result"]},
    ),
    tool_call("compacted"),
    ChatMessage.from_tool(
        tool_result="compacted" * 100,
        origin=ToolCall(tool_name="search", arguments={}, id="compacted"),
        meta={_COMPACTION_META_KEY: {"strategy": "other"}},
    ),
    tool_call("newest"),
    tool_result("newest", call_id="newest"),
]


class TestToolResultOffloadCompactor:
    def test_offloads_older_results(self, tmp_path):
        messages = conversation("a" * 400, "newest")
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=5).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        result = compacted[2].tool_call_result
        assert result is not None
        reference = compacted[2].meta["tool_result_offloaded"][0]
        assert result.result == f"Tool result offloaded to text (400 characters) at '{reference}'. Preview: aaaaa..."
        assert store.read(reference) == "a" * 400
        assert Path(reference).name == "compacted_search_c0.txt"
        assert compacted[2].meta[_COMPACTION_META_KEY] == {
            "strategy": "tool_result_offloading",
            "original_tokens": COUNTER.count(messages=[messages[2]]),
        }

    def test_offloads_binary_results(self, tmp_path):
        messages = [
            ChatMessage.from_user("task"),
            *_step([ImageContent(base64_image=base64.b64encode(PNG_BYTES).decode(), mime_type="image/png")], "old"),
            *conversation("newest")[1:],
        ]
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=PayloadCounter(chars_per_token=1)
        )

        assert compacted is not None
        assert store.read(compacted[2].meta["tool_result_offloaded"][0]) == PNG_BYTES
        assert [Path(path).name for path in tmp_path.iterdir()] == ["compacted_search_old.png"]

    def test_skips_ineligible_results(self, tmp_path):
        compacted = ToolResultOffloadCompactor(store=FileSystemToolResultStore(root=tmp_path), min_tokens=100).compact(
            messages=INELIGIBLE_CONVERSATION, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is None
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_keep_steps": 0}, "`min_keep_steps` must be at least 1"),
            ({"min_tokens": -1}, "`min_tokens` must be at least 0"),
            ({"preview_chars": -1}, "`preview_chars` must be at least 0"),
        ],
    )
    def test_rejects_invalid_settings(self, tmp_path, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ToolResultOffloadCompactor(store=FileSystemToolResultStore(root=tmp_path), **kwargs)

    def test_serialization_round_trip(self, tmp_path):
        hook = CompactionHook(
            compactor=ToolResultOffloadCompactor(
                store=FileSystemToolResultStore(root=tmp_path), min_keep_steps=2, min_tokens=12, preview_chars=42
            ),
            context_window=10_000,
        )
        restored = CompactionHook.from_dict(data=hook.to_dict()).compactor

        assert isinstance(restored, ToolResultOffloadCompactor)
        assert isinstance(restored.store, FileSystemToolResultStore)
        assert restored.store.root == tmp_path
        assert restored.min_keep_steps == 2
        assert restored.min_tokens == 12
        assert restored.preview_chars == 42


class TestToolResultOffloadCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self, tmp_path):
        messages = conversation("old" * 200, "newest")
        compactor = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        )
        compacted = await compactor.compact_async(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is not None
        assert compacted == compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER)
