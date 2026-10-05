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
from haystack.hooks.tool_result_offloading import FileSystemToolResultStore, ToolResultStore
from haystack.hooks.tool_result_offloading.utils import _content_block_payload
from haystack.token_counters.utils import _rendered_conversation
from haystack.tools import ToolsType
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COUNTER = FakeCounter(chars_per_token=1)

# A 1x1 PNG and a minimal PDF, both padded so their payloads clearly outweigh the reference replacing them.
PNG_BYTES = (
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    + b"\x00" * 1000
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n" + b"\xde\xad\xbe\xef" * 250 + b"\n%%EOF\n"


def _payload_placeholder(content: ChatMessageContentT) -> str:
    """Render an image or a file as its base64 payload."""
    assert isinstance(content, (TextContent, ImageContent, FileContent))
    return _content_block_payload(content)


class PayloadCounter(FakeCounter):
    """A counter that measures an image or a file by its base64 payload, the way a provider-side counter would."""

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        return len(_rendered_conversation(messages, placeholder=_payload_placeholder)) // self.chars_per_token


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


def _conversation(*results: str) -> list[ChatMessage]:
    """A user task followed by one Agent step per given result."""
    messages = [ChatMessage.from_user("task")]
    for index, result in enumerate(results):
        call_id = f"c{index}"
        messages.extend([tool_call(call_id), tool_result(result, call_id=call_id)])
    return messages


def _step(content: list[TextContent | ImageContent | FileContent], call_id: str) -> list[ChatMessage]:
    """An Agent step whose single tool result carries the given content blocks."""
    call = tool_call(call_id)
    return [call, ChatMessage.from_tool(tool_result=content, origin=call.tool_calls[0])]


FITTING_CONVERSATION = _conversation("a" * 400, "newest")
SINGLE_STEP_CONVERSATION = _conversation("only result")
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
        # The newest step calls two tools in parallel, so both of its results are protected together.
        messages = [
            *_conversation("a" * 400, "b" * 400),
            tool_call("parallel-1", "parallel-2"),
            tool_result("first" * 200, call_id="parallel-1"),
            tool_result("second" * 200, call_id="parallel-2"),
        ]
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=5).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        for index, original, preview in ((2, "a" * 400, "aaaaa"), (4, "b" * 400, "bbbbb")):
            result = compacted[index].tool_call_result
            assert result is not None
            reference = compacted[index].meta["tool_result_offloaded"][0]
            assert result.result == (
                f"Tool result offloaded to text (400 characters) at '{reference}'. Preview: {preview}..."
            )
            assert store.read(reference) == original
            assert compacted[index].meta[_COMPACTION_META_KEY] == {
                "strategy": "tool_result_offloading",
                "original_tokens": COUNTER.count(messages=[messages[index]]),
            }
        assert [Path(path).name for path in sorted(tmp_path.iterdir())] == [
            "compacted_search_c0.txt",
            "compacted_search_c1.txt",
        ]

        # The most recent step is left directly in context, and the caller-owned input list is unchanged.
        assert compacted[5:] == messages[5:]
        original_results = [message.tool_call_result for message in messages[2::2]]
        assert [result.result for result in original_results if result] == ["a" * 400, "b" * 400, "first" * 200]

    def test_stops_after_reaching_target(self, tmp_path):
        messages = _conversation("a" * 400, "b" * 400, "newest")
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        ).compact(messages=messages, target_tokens=COUNTER.count(messages=messages) - 1, token_counter=COUNTER)

        assert compacted is not None
        assert "tool_result_offloaded" in compacted[2].meta
        assert compacted[4] == messages[4]
        assert len(list(tmp_path.iterdir())) == 1

    def test_offloads_each_content_block(self, tmp_path):
        messages = [
            ChatMessage.from_user("task"),
            *_step([TextContent("A" * 2000), ImageContent(base64_image=base64.b64encode(PNG_BYTES).decode())], "old"),
            *_step([FileContent(base64_data=base64.b64encode(PDF_BYTES).decode(), mime_type="application/pdf")], "new"),
            *_conversation("newest")[1:],
        ]
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=0).compact(
            messages=messages, target_tokens=1, token_counter=PayloadCounter(chars_per_token=1)
        )

        assert compacted is not None
        text_reference, image_reference = compacted[2].meta["tool_result_offloaded"]
        assert store.read(text_reference) == "A" * 2000
        assert store.read(image_reference) == PNG_BYTES
        pointer = compacted[2].tool_call_result
        assert pointer is not None and isinstance(pointer.result, str)
        assert pointer.result.startswith("Tool result offloaded to 2 files:")
        assert store.read(compacted[4].meta["tool_result_offloaded"][0]) == PDF_BYTES
        assert [Path(path).name for path in sorted(tmp_path.iterdir())] == [
            "compacted_search_new.pdf",
            "compacted_search_old_0.txt",
            "compacted_search_old_1.png",
        ]

    def test_text_only_store_keeps_binary_results(self, caplog):
        messages = [
            ChatMessage.from_user("task"),
            *_step([ImageContent(base64_image=base64.b64encode(PNG_BYTES).decode())], "old"),
            *_conversation("newest")[1:],
        ]
        store = TextOnlyToolResultStore()
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=0).compact(
            messages=messages, target_tokens=1, token_counter=PayloadCounter(chars_per_token=1)
        )

        assert compacted is None
        assert not store.data
        assert "does not support binary content" in caplog.text

    @pytest.mark.parametrize(
        ("messages", "target_tokens", "min_tokens"),
        [
            pytest.param(
                FITTING_CONVERSATION, COUNTER.count(messages=FITTING_CONVERSATION), 0, id="conversation_already_fits"
            ),
            pytest.param(SINGLE_STEP_CONVERSATION, 1, 0, id="every_step_is_protected"),
            pytest.param(INELIGIBLE_CONVERSATION, 1, 100, id="no_eligible_results"),
        ],
    )
    def test_returns_none(self, tmp_path, messages, target_tokens, min_tokens):
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=min_tokens
        ).compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)

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
        messages = _conversation("old" * 200, "newest")
        compactor = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        )
        compacted = await compactor.compact_async(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is not None
        assert compacted == compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER)
