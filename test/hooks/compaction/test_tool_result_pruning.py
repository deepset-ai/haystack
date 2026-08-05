# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from haystack.dataclasses import ChatMessage, ImageContent
from haystack.hooks.compaction import ContextCompactionHook, ToolResultPruningCompactor
from haystack.hooks.compaction.tool_result_pruning import _DEFAULT_PLACEHOLDER
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.token_counters import ApproximateTokenCounter
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COUNTER = FakeCounter(chars_per_token=1)


def _conversation(*results: str) -> list[ChatMessage]:
    messages = [ChatMessage.from_user("task")]
    for index, result in enumerate(results):
        call_id = f"c{index}"
        messages.extend([tool_call(call_id), tool_result(result, call_id=call_id)])
    return messages


class TestToolResultPruningCompactor:
    def test_prunes_old_results_until_the_target_is_reached(self):
        messages = _conversation("a" * 400, "b" * 400, "c" * 400)
        compactor = ToolResultPruningCompactor(min_keep_results=1, min_tokens=0)
        one_pruned = list(messages)
        replacement = compactor._prune(message=messages[2], token_counter=COUNTER)
        assert replacement is not None
        pruned, _ = replacement
        one_pruned[2] = pruned
        target = COUNTER.count(messages=one_pruned)

        compacted = compactor.compact(messages=messages, target_tokens=target, token_counter=COUNTER)

        assert compacted is not None
        assert compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == _DEFAULT_PLACEHOLDER.replace("{tool_name}", "search")
        assert compacted[4:] == messages[4:]
        assert messages[2].tool_call_result is not None
        assert messages[2].tool_call_result.result == "a" * 400

    def test_counts_the_full_conversation_only_once(self):
        messages = _conversation("a" * 400, "b" * 400, "c" * 400)
        counter = Mock(wraps=COUNTER)

        compacted = ToolResultPruningCompactor(min_keep_results=1, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=counter
        )

        assert compacted is not None
        counted_message_lengths = [len(call.kwargs["messages"]) for call in counter.count.call_args_list]
        assert counted_message_lengths == [len(messages), 1, 1, 1, 1]

    def test_keeps_the_minimum_number_of_recent_results(self):
        messages = _conversation("a" * 400, "b" * 400, "c" * 400)

        compacted = ToolResultPruningCompactor(min_keep_results=2, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert compacted[2] != messages[2]
        assert compacted[4:] == messages[4:]

    def test_keeps_the_entire_fresh_parallel_result_batch(self):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("old"),
            tool_result("old" * 200, call_id="old"),
            tool_call("fresh-1", "fresh-2", "fresh-3"),
            tool_result("first" * 200, call_id="fresh-1"),
            tool_result("second" * 200, call_id="fresh-2"),
            tool_result("third" * 200, call_id="fresh-3"),
        ]

        compacted = ToolResultPruningCompactor(min_keep_results=1, min_tokens=0).compact(
            messages=messages, target_tokens=1, token_counter=COUNTER
        )

        assert compacted is not None
        assert compacted[2] != messages[2]
        assert compacted[3:] == messages[3:]

    def test_returns_none_when_the_conversation_already_fits(self):
        messages = _conversation("a" * 400, "b" * 400)

        assert (
            ToolResultPruningCompactor(min_keep_results=1, min_tokens=0).compact(
                messages=messages, target_tokens=COUNTER.count(messages=messages), token_counter=COUNTER
            )
            is None
        )

    def test_returns_none_when_there_are_no_old_results(self):
        messages = _conversation("only result")

        assert (
            ToolResultPruningCompactor(min_keep_results=1, min_tokens=0).compact(
                messages=messages, target_tokens=1, token_counter=COUNTER
            )
            is None
        )

    def test_returns_none_when_the_placeholder_would_not_save_tokens(self):
        messages = _conversation("short", "newest")
        compactor = ToolResultPruningCompactor(
            min_keep_results=1, min_tokens=0, placeholder="a placeholder much longer than the result"
        )

        assert compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER) is None

    def test_min_tokens_accounts_for_non_text_tool_results(self):
        image = ImageContent(base64_image="Zm9v", mime_type="image/png")
        image_call = tool_call("image")
        messages = [
            ChatMessage.from_user("task"),
            image_call,
            ChatMessage.from_tool(tool_result=[image], origin=image_call.tool_calls[0]),
            tool_call("newest"),
            tool_result("newest", call_id="newest"),
        ]
        counter = ApproximateTokenCounter(tokens_per_image=500)

        compacted = ToolResultPruningCompactor(min_keep_results=1, min_tokens=100).compact(
            messages=messages, target_tokens=1, token_counter=counter
        )

        assert compacted is not None and compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == _DEFAULT_PLACEHOLDER.replace("{tool_name}", "search")

    def test_skips_small_error_offloaded_and_previously_compacted_results(self):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("small"),
            tool_result("small", call_id="small"),
            tool_call("error"),
            tool_result("x" * 400, call_id="error", error=True),
            tool_call("offloaded"),
            ChatMessage.from_tool(
                tool_result="x" * 400,
                origin=tool_call("offloaded").tool_calls[0],
                meta={"tool_result_offloaded": "result.txt"},
            ),
            tool_call("compacted"),
            ChatMessage.from_tool(
                tool_result="x" * 400,
                origin=tool_call("compacted").tool_calls[0],
                meta={_COMPACTION_META_KEY: {"strategy": "other"}},
            ),
            tool_call("newest"),
            tool_result("x" * 400, call_id="newest"),
        ]

        assert (
            ToolResultPruningCompactor(min_keep_results=1).compact(
                messages=messages, target_tokens=1, token_counter=COUNTER
            )
            is None
        )

    def test_preserves_origin_error_and_meta_and_records_compaction(self):
        messages = _conversation("result " * 100, "newest")
        messages[2].meta["custom"] = "value"

        compacted = ToolResultPruningCompactor(min_keep_results=1, min_tokens=0).compact(
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

    def test_custom_placeholder_replaces_tool_name_and_preserves_literal_braces(self):
        messages = _conversation("abcdefghij" * 40, "newest")
        compactor = ToolResultPruningCompactor(
            min_keep_results=1, min_tokens=0, placeholder='Run {tool_name} again with {"query": "..."}.'
        )

        compacted = compactor.compact(messages=messages, target_tokens=1, token_counter=COUNTER)

        assert compacted is not None and compacted[2].tool_call_result is not None
        assert compacted[2].tool_call_result.result == 'Run search again with {"query": "..."}.'

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_keep_results": 0}, "`min_keep_results` must be at least 1"),
            ({"min_tokens": -1}, "`min_tokens` must be at least 0"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ToolResultPruningCompactor(**kwargs)

    def test_serialization_round_trip(self):
        compactor = ToolResultPruningCompactor(
            min_keep_results=2, min_tokens=12, placeholder="", skip_meta_keys=("stored", "cached")
        )

        restored = ToolResultPruningCompactor.from_dict(data=compactor.to_dict())

        assert isinstance(restored, ToolResultPruningCompactor)
        assert restored.min_keep_results == 2
        assert restored.min_tokens == 12
        assert restored.placeholder == ""
        assert restored.skip_meta_keys == ("stored", "cached")

    def test_survives_a_hook_serialization_round_trip(self):
        hook = ContextCompactionHook(compactor=ToolResultPruningCompactor(min_keep_results=2), context_window=10_000)

        restored = ContextCompactionHook.from_dict(data=hook.to_dict())

        assert isinstance(restored.compactor, ToolResultPruningCompactor)
        assert restored.compactor.min_keep_results == 2
