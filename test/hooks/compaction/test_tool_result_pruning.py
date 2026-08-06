# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ImageContent
from haystack.hooks.compaction import CompactionHook, ToolResultPruningCompactor
from haystack.hooks.compaction.tool_result_pruning import _DEFAULT_PLACEHOLDER
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.hooks.tool_result_offloading import AlwaysOffload, FileSystemToolResultStore, ToolResultOffloadHook
from haystack.token_counters import ApproximateTokenCounter
from test.hooks.compaction.helpers import FakeCounter, make_state, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COUNTER = FakeCounter(chars_per_token=1)


def _conversation(*results: str) -> list[ChatMessage]:
    messages = [ChatMessage.from_user("task")]
    for index, result in enumerate(results):
        call_id = f"c{index}"
        messages.extend([tool_call(call_id), tool_result(result, call_id=call_id)])
    return messages


class TestToolResultPruningCompactor:
    def test_stops_pruning_after_reaching_target(self):
        # Conversation with seven messages: one user message followed by three tool-call/result pairs.
        messages = _conversation("a" * 400, "b" * 400, "c" * 400)
        compactor = ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0)
        # We pre-calculated the target token count to just be enough to only remove the oldest tool result based on the
        # count from the FakeCounter.
        target_tokens = 1041
        compacted = compactor.compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)
        assert compacted is not None
        assert compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == _DEFAULT_PLACEHOLDER.replace("{tool_name}", "search")
        assert compacted[4:] == messages[4:]
        # Ensure compaction returns new messages instead of changing the input.
        assert [messages[index].tool_call_result.result for index in (2, 4, 6)] == ["a" * 400, "b" * 400, "c" * 400]

    def test_prunes_multiple_results_in_one_call(self):
        messages = _conversation("a" * 400, "b" * 400, "newest")
        compacted = ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        for index in (2, 4):
            assert compacted[index].tool_call_result is not None
            assert compacted[index].tool_call_result.result == _DEFAULT_PLACEHOLDER.replace("{tool_name}", "search")
        assert compacted[5:] == messages[5:]

    def test_keeps_min_keep_steps(self):
        # Three tool-calling steps, with three parallel results in the middle step.
        messages = [
            ChatMessage.from_user("task"),
            tool_call("old"),
            tool_result("old" * 200, call_id="old"),
            tool_call("parallel-1", "parallel-2", "parallel-3"),
            tool_result("first" * 200, call_id="parallel-1"),
            tool_result("second" * 200, call_id="parallel-2"),
            tool_result("third" * 200, call_id="parallel-3"),
            tool_call("newest"),
            tool_result("newest" * 200, call_id="newest"),
        ]
        # Keeping two steps protects the complete parallel step and the newest step, leaving only `old` eligible.
        compacted = ToolResultPruningCompactor(min_keep_steps=2, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        assert compacted[2] != messages[2]
        assert compacted[3:] == messages[3:]

    def test_returns_none_when_the_conversation_already_fits(self):
        messages = _conversation("a" * 400, "b" * 400)
        assert (
            ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
                messages=messages, target_tokens=COUNTER.count(messages=messages), token_counter=COUNTER
            )
            is None
        )

    def test_returns_none_when_all_steps_are_protected(self):
        messages = _conversation("only result")
        assert (
            ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
                messages=messages, target_tokens=1, token_counter=COUNTER
            )
            is None
        )

    def test_returns_none_when_the_placeholder_would_not_save_tokens(self):
        messages = _conversation("short", "newest")
        compactor = ToolResultPruningCompactor(
            min_keep_steps=1, min_tokens=0, placeholder="a placeholder much longer than the result"
        )
        assert compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER) is None

    def test_compact_with_image_result(self):
        image = ImageContent(base64_image="Zm9v", mime_type="image/png")
        image_call = tool_call("image", name="image_generator")
        messages = [
            ChatMessage.from_user("task"),
            image_call,
            ChatMessage.from_tool(tool_result=[image], origin=image_call.tool_calls[0]),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        # The tiny payload exceeds `min_tokens` only if the counter includes its configured per-image token cost.
        counter = ApproximateTokenCounter(tokens_per_image=500)
        compacted = ToolResultPruningCompactor(min_keep_steps=1, min_tokens=100).compact(
            messages=messages, target_tokens=1, token_counter=counter
        )
        assert compacted is not None and compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == _DEFAULT_PLACEHOLDER.replace("{tool_name}", "image_generator")

    def test_skips_results_below_min_tokens(self):
        messages = _conversation("small", "newest")
        compacted = ToolResultPruningCompactor(min_keep_steps=1, min_tokens=200).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is None

    def test_skips_previously_compacted_results(self):
        compacted_call = tool_call("compacted")
        messages = [
            ChatMessage.from_user("task"),
            compacted_call,
            ChatMessage.from_tool(
                tool_result="x" * 400,
                origin=compacted_call.tool_calls[0],
                meta={_COMPACTION_META_KEY: {"strategy": "other"}},
            ),
            tool_call("newest"),
            tool_result("x" * 400, call_id="newest"),
        ]
        # With target_tokens=1 and min_tokens=0, the compaction marker is the only reason the old result is skipped.
        assert (
            ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
                messages=messages, target_tokens=1, token_counter=COUNTER
            )
            is None
        )

    def test_skips_offloaded_result(self, tmp_path):
        messages = [ChatMessage.from_user("task"), tool_call("offloaded"), tool_result("x" * 400, call_id="offloaded")]
        state = make_state(messages)
        # The after_tool hook only rewrites the trailing batch, which represents results from the current Agent step.
        offload_hook = ToolResultOffloadHook(
            store=FileSystemToolResultStore(root=tmp_path), offload_strategies={"*": AlwaysOffload()}
        )
        offload_hook.run(state)
        offloaded_result = state.data["messages"][-1]
        assert offloaded_result.tool_call_result is not None
        assert offloaded_result.tool_call_result.result.startswith("Tool result offloaded")
        assert "tool_result_offloaded" in offloaded_result.meta
        # A subsequent Agent step turns the offloaded pointer into an old result that pruning would normally consider.
        history = [*state.data["messages"], tool_call("newest"), tool_result("newest", call_id="newest")]
        # Disable the size threshold so only the offloading marker prevents the old pointer from being pruned.
        assert (
            ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
                messages=history, target_tokens=1, token_counter=COUNTER
            )
            is None
        )

    def test_pruned_result_preserves_properties_and_records_metadata(self):
        failed_call = tool_call("failed", name="failing_tool")
        messages = [
            ChatMessage.from_user("task"),
            failed_call,
            ChatMessage.from_tool(
                tool_result="error details " * 100,
                origin=failed_call.tool_calls[0],
                error=True,
                meta={"custom": "value"},
            ),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        # Old errors are prunable, but their origin, error state, and existing metadata must survive the rewrite.
        compacted = ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )
        assert compacted is not None
        original_result = messages[2].tool_call_result
        pruned_result = compacted[2].tool_call_result
        assert original_result is not None and pruned_result is not None
        assert pruned_result.origin == original_result.origin
        assert pruned_result.error == original_result.error
        assert compacted[2].meta == {
            "custom": "value",
            _COMPACTION_META_KEY: {
                "strategy": "tool_result_pruning",
                "original_tokens": COUNTER.count(messages=[messages[2]]),
            },
        }

    def test_custom_placeholder(self):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("old", name="database_query"),
            tool_result("abcdefghij" * 40, call_id="old", name="database_query"),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        compactor = ToolResultPruningCompactor(
            min_keep_steps=1, min_tokens=0, placeholder='Run {tool_name} again with {"query": "..."}.'
        )
        compacted = compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER)
        assert compacted is not None and compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == 'Run database_query again with {"query": "..."}.'

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_keep_steps": 0}, "`min_keep_steps` must be at least 1"),
            ({"min_tokens": -1}, "`min_tokens` must be at least 0"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ToolResultPruningCompactor(**kwargs)

    def test_serialization_round_trip(self):
        compactor = ToolResultPruningCompactor(
            min_keep_steps=2, min_tokens=12, placeholder="", skip_meta_keys=("stored", "cached")
        )
        restored = ToolResultPruningCompactor.from_dict(data=compactor.to_dict())
        assert isinstance(restored, ToolResultPruningCompactor)
        assert restored.min_keep_steps == 2
        assert restored.min_tokens == 12
        assert restored.placeholder == ""
        assert restored.skip_meta_keys == ("stored", "cached")

    def test_hook_serialization_round_trip(self):
        hook = CompactionHook(compactor=ToolResultPruningCompactor(min_keep_steps=2), context_window=10_000)
        restored = CompactionHook.from_dict(data=hook.to_dict())
        assert isinstance(restored.compactor, ToolResultPruningCompactor)
        assert restored.compactor.min_keep_steps == 2
