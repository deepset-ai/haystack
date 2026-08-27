# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from haystack.dataclasses import ChatMessage, ImageContent, TextContent
from haystack.hooks.compaction import CompactionHook, ToolResultOffloadCompactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.hooks.tool_result_offloading import FileSystemToolResultStore
from test.hooks.compaction.helpers import FakeCounter, make_state, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COUNTER = FakeCounter(chars_per_token=1)


def _conversation(*results: str) -> list[ChatMessage]:
    messages = [ChatMessage.from_user("task")]
    for index, result in enumerate(results):
        call_id = f"c{index}"
        messages.extend([tool_call(call_id), tool_result(result, call_id=call_id)])
    return messages


class TestToolResultOffloadCompactor:
    def test_offloads_old_results_and_keeps_latest_step(self, tmp_path):
        messages = _conversation("a" * 400, "b" * 400, "newest")
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_keep_steps=1, min_tokens=0, preview_chars=5).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        for index, original in ((2, "a" * 400), (4, "b" * 400)):
            result = compacted[index].tool_call_result
            assert result is not None
            assert isinstance(result.result, str)
            assert result.result.startswith("Tool result offloaded")
            assert "400 characters" in result.result
            assert "aaaaa..." in result.result or "bbbbb..." in result.result
            assert store.read(compacted[index].meta["tool_result_offloaded"]) == original
            assert compacted[index].meta[_COMPACTION_META_KEY]["strategy"] == "tool_result_offloading"

        assert compacted[5:] == messages[5:]
        # The compactor must leave the caller-owned input unchanged.
        original_results = [messages[index].tool_call_result for index in (2, 4, 6)]
        assert all(result is not None for result in original_results)
        assert [result.result for result in original_results if result is not None] == ["a" * 400, "b" * 400, "newest"]

    def test_keeps_all_results_from_protected_parallel_step(self, tmp_path):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("old"),
            tool_result("old" * 200, call_id="old"),
            tool_call("parallel-1", "parallel-2"),
            tool_result("first" * 200, call_id="parallel-1"),
            tool_result("second" * 200, call_id="parallel-2"),
        ]
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_keep_steps=1, min_tokens=0, preview_chars=0
        ).compact(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is not None
        assert "tool_result_offloaded" in compacted[2].meta
        assert compacted[3:] == messages[3:]

    def test_stops_offloading_after_reaching_target(self, tmp_path):
        messages = _conversation("a" * 400, "b" * 400, "newest")
        current_tokens = COUNTER.count(messages=messages)
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        ).compact(messages=messages, target_tokens=current_tokens - 1, token_counter=COUNTER)

        assert compacted is not None
        assert "tool_result_offloaded" in compacted[2].meta
        assert compacted[4] == messages[4]
        assert len(list(Path(tmp_path).iterdir())) == 1

    def test_returns_none_when_conversation_already_fits(self, tmp_path):
        messages = _conversation("a" * 400, "newest")
        compacted = ToolResultOffloadCompactor(store=FileSystemToolResultStore(root=tmp_path)).compact(
            messages=messages, target_tokens=COUNTER.count(messages=messages), token_counter=COUNTER
        )
        assert compacted is None
        assert not list(Path(tmp_path).iterdir())

    def test_returns_none_when_all_steps_are_protected(self, tmp_path):
        messages = _conversation("only result")
        compacted = ToolResultOffloadCompactor(store=FileSystemToolResultStore(root=tmp_path), min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is None
        assert not list(Path(tmp_path).iterdir())

    def test_skips_small_error_and_previously_rewritten_results(self, tmp_path):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("small"),
            tool_result("small", call_id="small"),
            tool_call("error"),
            tool_result("error" * 100, call_id="error", error=True),
            tool_call("offloaded"),
            ChatMessage.from_tool(
                tool_result="offloaded" * 100,
                origin=tool_call("offloaded").tool_calls[0],
                meta={"tool_result_offloaded": "stored-result"},
            ),
            tool_call("compacted"),
            ChatMessage.from_tool(
                tool_result="compacted" * 100,
                origin=tool_call("compacted").tool_calls[0],
                meta={_COMPACTION_META_KEY: {"strategy": "other"}},
            ),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=100, preview_chars=0
        ).compact(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is None
        assert not list(Path(tmp_path).iterdir())

    def test_non_text_result_is_not_offloaded(self, tmp_path, caplog):
        image_call = tool_call("image", name="image_generator")
        messages = [
            ChatMessage.from_user("task"),
            image_call,
            ChatMessage.from_tool(
                tool_result=[ImageContent(base64_image="Zm9v", mime_type="image/png")], origin=image_call.tool_calls[0]
            ),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        compacted = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        ).compact(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is None
        assert "produced a non-text result" in caplog.text
        assert not list(Path(tmp_path).iterdir())

    def test_offloads_text_content_sequence(self, tmp_path):
        old_call = tool_call("old")
        messages = [
            ChatMessage.from_user("task"),
            old_call,
            ChatMessage.from_tool(
                tool_result=[TextContent("A" * 200), TextContent("B" * 200)], origin=old_call.tool_calls[0]
            ),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        store = FileSystemToolResultStore(root=tmp_path)
        compacted = ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert store.read(compacted[2].meta["tool_result_offloaded"]) == "A" * 200 + "B" * 200

    @pytest.mark.asyncio
    async def test_compact_async(self, tmp_path):
        messages = _conversation("old" * 200, "newest")
        compacted = await ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_tokens=0, preview_chars=0
        ).compact_async(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is not None
        assert "tool_result_offloaded" in compacted[2].meta

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
        compactor = ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root=tmp_path), min_keep_steps=2, min_tokens=12, preview_chars=42
        )
        restored = ToolResultOffloadCompactor.from_dict(data=compactor.to_dict())

        assert isinstance(restored, ToolResultOffloadCompactor)
        assert isinstance(restored.store, FileSystemToolResultStore)
        assert restored.store.root == tmp_path
        assert restored.min_keep_steps == 2
        assert restored.min_tokens == 12
        assert restored.preview_chars == 42

    def test_hook_serialization_round_trip(self, tmp_path):
        hook = CompactionHook(
            compactor=ToolResultOffloadCompactor(store=FileSystemToolResultStore(root=tmp_path), min_keep_steps=2),
            context_window=10_000,
        )
        restored = CompactionHook.from_dict(data=hook.to_dict())

        assert isinstance(restored.compactor, ToolResultOffloadCompactor)
        assert restored.compactor.min_keep_steps == 2


class TestToolResultOffloadCompactorInHook:
    def test_results_stay_inline_until_compaction_is_triggered(self, tmp_path):
        messages = _conversation("old" * 300, "newest" * 100)
        state = make_state(messages, context_tokens=0)
        store = FileSystemToolResultStore(root=tmp_path)
        hook = CompactionHook(
            compactor=ToolResultOffloadCompactor(store=store, min_tokens=0, preview_chars=0),
            context_window=2_000,
            compact_at=0.5,
            compact_to=0.2,
            token_counter=COUNTER,
        )

        assert state.data["messages"][2].tool_call_result.result == "old" * 300
        hook.run(state)

        offloaded = state.data["messages"][2]
        assert offloaded.tool_call_result is not None
        assert offloaded.tool_call_result.result.startswith("Tool result offloaded")
        assert store.read(offloaded.meta["tool_result_offloaded"]) == "old" * 300
        # The latest output is still fresh and remains directly available to the next LLM call.
        assert state.data["messages"][-1].tool_call_result.result == "newest" * 100
