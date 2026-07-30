# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction import SlidingWindowCompactor
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, count_markers, long_conversation, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# A target of one token forces the window down to `min_keep_messages`, isolating the structural rules from sizing.
SMALLEST = 1
# A fixed, obvious rate, so the tests that do exercise sizing can reason in round numbers.
COUNTER = FakeCounter()


def _marker(message: ChatMessage) -> dict[str, Any]:
    return message.meta[_COMPACTION_META_KEY]


class TestSlidingWindowCompactor:
    def test_replaces_the_middle_with_an_omission_note(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_messages=2).compact(messages, SMALLEST, COUNTER)
        assert compacted is not None
        # The system prefix survives, the note stands in for what was dropped, and the window is kept verbatim.
        assert compacted[0] is messages[0]
        assert compacted[2:] == messages[4:]
        assert _marker(compacted[1]) == {"strategy": "sliding_window", "removed_messages": 3, "kept_messages": 2}
        assert compacted[1].text == _DEFAULT_OMISSION_NOTE.replace("{num_removed}", "3")
        # A user message, not a system one, so providers that hoist system messages cannot move it out of position.
        assert compacted[1].is_from(ChatRole.USER)

    def test_a_roomier_target_keeps_more(self):
        messages = [ChatMessage.from_user("x" * 100) for _ in range(10)]
        compactor = SlidingWindowCompactor(min_keep_messages=1)

        tight = compactor.compact(messages, 60, COUNTER)
        roomy = compactor.compact(messages, 200, COUNTER)

        assert tight is not None
        assert roomy is not None
        assert len(roomy) > len(tight)

    def test_returns_none_when_the_conversation_already_fits(self):
        assert SlidingWindowCompactor(min_keep_messages=1).compact(long_conversation(), 100_000, COUNTER) is None

    def test_omission_note_can_be_turned_off(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_messages=2, omission_note=None).compact(messages, SMALLEST, COUNTER)
        assert compacted == [messages[0], *messages[4:]]

    @pytest.mark.parametrize(
        ("note", "expected"),
        [
            pytest.param("Dropped {num_removed} messages.", "Dropped 3 messages.", id="placeholder-substituted"),
            pytest.param("Some history is missing.", "Some history is missing.", id="no-placeholder-used-as-written"),
            # A note is arbitrary user text, so braces of its own must not be treated as placeholders.
            pytest.param("Gone: {other} x{num_removed}", "Gone: {other} x3", id="other-braces-left-alone"),
        ],
    )
    def test_omission_note_can_be_customized(self, note, expected):
        compacted = SlidingWindowCompactor(min_keep_messages=2, omission_note=note).compact(
            long_conversation(), SMALLEST, COUNTER
        )
        assert compacted is not None
        assert compacted[1].text == expected

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="empty"),
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], id="only-system"),
            pytest.param([ChatMessage.from_system("rules"), ChatMessage.from_user("hi")], id="nothing-outside-window"),
        ],
    )
    def test_returns_none_when_there_is_nothing_to_remove(self, messages):
        assert SlidingWindowCompactor(min_keep_messages=20).compact(messages, SMALLEST, COUNTER) is None

    def test_keeps_a_tool_call_together_with_its_results(self):
        # A one-message window would start at a tool result whose call had been dropped, which chat-completion APIs
        # reject. The window grows instead.
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_messages=1).compact(messages, SMALLEST, COUNTER)
        assert compacted is not None
        # 1 system, 1 note, then the call and its result kept together.
        assert len(compacted) == 4
        assert compacted[-2:] == messages[4:]
        assert compacted[-2].tool_calls[0].id == compacted[-1].tool_call_result.origin.id

    @pytest.mark.parametrize("min_keep_messages", [1, 2, 3, 20])
    def test_never_drops_the_final_message(self, min_keep_messages):
        # The Agent may be mid-step with a pending tool call whose result is appended right after compaction returns.
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_messages=min_keep_messages).compact(messages, SMALLEST, COUNTER)
        assert (compacted or messages)[-1] is messages[-1]

    def test_repeated_compaction_folds_the_previous_note(self):
        compactor = SlidingWindowCompactor(min_keep_messages=2)
        first = compactor.compact(long_conversation(), SMALLEST, COUNTER)
        assert first is not None

        # Simulate two more turns arriving on top of the already-compacted conversation.
        grown = [*first, tool_call("c3"), tool_result("third result", call_id="c3")]
        second = compactor.compact(grown, SMALLEST, COUNTER)

        assert second is not None
        # Exactly one note: the earlier one is inside the block that was dropped, not carried along beside the new one.
        assert count_markers(second) == 1
        assert second[0].text == "rules"

    def test_returns_none_when_the_note_would_cost_what_it_saves(self):
        # Four messages with a one-message system prefix: a two-message window leaves exactly one to remove, which a
        # note would merely replace.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("a"),
            ChatMessage.from_user("b"),
            ChatMessage.from_user("c"),
        ]
        assert SlidingWindowCompactor(min_keep_messages=2).compact(messages, SMALLEST, COUNTER) is None
        # Without a note there is a real saving, so the same cut is worth making.
        assert SlidingWindowCompactor(min_keep_messages=2, omission_note=None).compact(messages, SMALLEST, COUNTER) == [
            messages[0],
            *messages[2:],
        ]

    def test_rejects_keeping_no_messages(self):
        with pytest.raises(ValueError, match="`min_keep_messages` must be at least 1"):
            SlidingWindowCompactor(min_keep_messages=0)

    def test_serde_round_trip(self):
        compactor = SlidingWindowCompactor(min_keep_messages=7, omission_note="Dropped {num_removed}.")
        data = compactor.to_dict()
        assert data == {
            "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
            "init_parameters": {"min_keep_messages": 7, "omission_note": "Dropped {num_removed}."},
        }
        restored = SlidingWindowCompactor.from_dict(data)
        assert restored.min_keep_messages == 7
        assert restored.omission_note == "Dropped {num_removed}."


class TestSlidingWindowCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        # `SlidingWindowCompactor` does no I/O, so it relies on the protocol's default `compact_async`.
        compactor = SlidingWindowCompactor(min_keep_messages=2)
        messages = long_conversation()

        assert await compactor.compact_async(messages, SMALLEST, COUNTER) == compactor.compact(
            messages, SMALLEST, COUNTER
        )

    @pytest.mark.asyncio
    async def test_compact_async_returns_none_when_nothing_to_remove(self):
        compactor = SlidingWindowCompactor(min_keep_messages=20)
        assert await compactor.compact_async([ChatMessage.from_user("hi")], SMALLEST, COUNTER) is None
