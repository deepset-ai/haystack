# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction import SlidingWindowCompactor
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, count_markers, long_conversation, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# A target of one token forces the window down to `min_keep_steps`, isolating the structural rules from sizing.
SMALLEST = 1
# A fixed, obvious rate, so the tests that do exercise sizing can reason in round numbers.
COUNTER = FakeCounter()


class TestSlidingWindowCompactor:
    def test_replaces_the_middle_with_an_omission_note(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor().compact(messages, SMALLEST, COUNTER)
        assert compacted is not None
        # The instructions and task survive, the note stands in for the older step, and the latest step stays intact.
        assert compacted[:2] == messages[:2]
        assert compacted[3:] == messages[4:]
        assert compacted[2].meta[_COMPACTION_META_KEY] == {
            "strategy": "sliding_window",
            "removed_messages": 2,
            "kept_messages": 4,
            "kept_steps": 1,
        }
        assert compacted[2].text == _DEFAULT_OMISSION_NOTE.replace("{num_removed}", "2")
        # A user message, not a system one, so providers that hoist system messages cannot move it out of position.
        assert compacted[2].is_from(ChatRole.USER)

    def test_a_roomier_target_keeps_more(self):
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_user("task")]
        for index in range(4):
            messages.extend([tool_call(f"c{index}"), tool_result("x" * 400, call_id=f"c{index}")])
        compactor = SlidingWindowCompactor(omission_note=None)

        tight = compactor.compact(messages, 60, COUNTER)
        roomy = compactor.compact(messages, 350, COUNTER)

        assert tight is not None
        assert roomy is not None
        assert sum(message.is_from(ChatRole.ASSISTANT) for message in roomy) > sum(
            message.is_from(ChatRole.ASSISTANT) for message in tight
        )

    def test_returns_none_when_the_conversation_already_fits(self):
        assert SlidingWindowCompactor().compact(long_conversation(), 100_000, COUNTER) is None

    def test_omission_note_can_be_turned_off(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(omission_note=None).compact(messages, SMALLEST, COUNTER)
        assert compacted == [*messages[:2], *messages[4:]]

    @pytest.mark.parametrize(
        ("note", "expected"),
        [
            pytest.param("Dropped {num_removed} messages.", "Dropped 2 messages.", id="placeholder-substituted"),
            pytest.param("Some history is missing.", "Some history is missing.", id="no-placeholder-used-as-written"),
            # A note is arbitrary user text, so braces of its own must not be treated as placeholders.
            pytest.param("Gone: {other} x{num_removed}", "Gone: {other} x2", id="other-braces-left-alone"),
        ],
    )
    def test_omission_note_can_be_customized(self, note, expected):
        compacted = SlidingWindowCompactor(omission_note=note).compact(long_conversation(), SMALLEST, COUNTER)
        assert compacted is not None
        assert compacted[2].text == expected

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="empty"),
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], id="only-system"),
            pytest.param([ChatMessage.from_system("rules"), ChatMessage.from_user("hi")], id="nothing-outside-window"),
        ],
    )
    def test_returns_none_when_there_is_nothing_to_remove(self, messages):
        assert SlidingWindowCompactor().compact(messages, SMALLEST, COUNTER) is None

    def test_keeps_a_parallel_tool_call_together_with_all_results(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("task"),
            tool_call("old"),
            tool_result("old result " * 40, call_id="old"),
            tool_call("a", "b", "c"),
            tool_result("first", call_id="a"),
            tool_result("second", call_id="b"),
            tool_result("third", call_id="c"),
        ]

        compacted = SlidingWindowCompactor().compact(messages, SMALLEST, COUNTER)

        assert compacted is not None
        assert compacted[-4:] == messages[-4:]
        assert {call.id for call in compacted[-4].tool_calls} == {
            result.tool_call_result.origin.id for result in compacted[-3:]
        }

    @pytest.mark.parametrize(("min_keep_steps", "expected"), [(0, 0), (1, 1), (2, 2), (20, 2)])
    def test_min_keep_steps_wins_over_an_unaffordable_target(self, min_keep_steps, expected):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_steps=min_keep_steps, omission_note=None).compact(
            messages, SMALLEST, COUNTER
        )

        result = compacted or messages
        assert sum(message.is_from(ChatRole.ASSISTANT) for message in result) == expected

    def test_rejects_negative_min_keep_steps(self):
        with pytest.raises(ValueError, match="`min_keep_steps` must be at least 0"):
            SlidingWindowCompactor(min_keep_steps=-1)

    def test_preserves_the_latest_user_task_when_compacting_input_history(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question " * 100),
            ChatMessage.from_assistant("old answer"),
            ChatMessage.from_user("current task"),
        ]

        compacted = SlidingWindowCompactor(min_keep_steps=0, omission_note=None).compact(messages, SMALLEST, COUNTER)

        assert compacted == [messages[0], messages[-1]]

    def test_repeated_compaction_folds_the_previous_note(self):
        compactor = SlidingWindowCompactor()
        first = compactor.compact(long_conversation(), SMALLEST, COUNTER)
        assert first is not None

        # Simulate two more turns arriving on top of the already-compacted conversation.
        grown = [*first, tool_call("c3"), tool_result("third result", call_id="c3")]
        second = compactor.compact(grown, SMALLEST, COUNTER)

        assert second is not None
        # The original task stays anchored and the earlier compaction note is folded into the new one.
        assert count_markers(second) == 1
        assert second[0].text == "rules"
        assert second[1].text == "start"

    @pytest.mark.parametrize(
        ("removable", "worth_replacing"),
        [
            pytest.param("a", False, id="cheaper-than-the-note"),
            pytest.param("x" * 4000, True, id="dearer-than-the-note"),
        ],
    )
    def test_a_cut_is_made_only_when_the_note_costs_less_than_what_it_replaces(self, removable, worth_replacing):
        # A note is not free, so what matters is the size of what goes rather than how many messages it is: one long
        # tool result is worth replacing, one short message is not.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("task"),
            ChatMessage.from_assistant(removable),
            ChatMessage.from_assistant("latest step"),
        ]
        compacted = SlidingWindowCompactor().compact(messages, SMALLEST, COUNTER)

        assert (compacted is not None) is worth_replacing
        # Without a note there is nothing to pay for, so the same cut is always worth making.
        assert SlidingWindowCompactor(omission_note=None).compact(messages, SMALLEST, COUNTER) == [
            *messages[:2],
            messages[-1],
        ]

    def test_keeping_no_steps_still_preserves_the_current_task(self):
        messages = long_conversation()

        compacted = SlidingWindowCompactor(min_keep_steps=0, omission_note=None).compact(messages, SMALLEST, COUNTER)

        assert compacted == messages[:2]

    def test_serde_round_trip(self):
        compactor = SlidingWindowCompactor(
            min_keep_steps=7, preserve_last_user_message=False, omission_note="Dropped {num_removed}."
        )
        data = compactor.to_dict()
        assert data == {
            "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
            "init_parameters": {
                "min_keep_steps": 7,
                "preserve_last_user_message": False,
                "omission_note": "Dropped {num_removed}.",
            },
        }
        restored = SlidingWindowCompactor.from_dict(data)
        assert restored.min_keep_steps == 7
        assert restored.preserve_last_user_message is False
        assert restored.omission_note == "Dropped {num_removed}."


class TestSlidingWindowCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        # `SlidingWindowCompactor` does no I/O, so it relies on the protocol's default `compact_async`.
        compactor = SlidingWindowCompactor()
        messages = long_conversation()

        assert await compactor.compact_async(messages, SMALLEST, COUNTER) == compactor.compact(
            messages, SMALLEST, COUNTER
        )
