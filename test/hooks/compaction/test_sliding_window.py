# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction import SlidingWindowCompactor
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE, _historical_turn_spans
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import FakeCounter, count_markers, long_conversation, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# A target of one token forces the window down to `min_keep_steps`, isolating the structural rules from sizing.
SMALLEST = 1
# A fixed, obvious rate, so the tests that do exercise sizing can reason in round numbers.
COUNTER = FakeCounter()


class TestHistoricalTurnSpans:
    def test_groups_each_user_message_with_its_assistant_steps_and_tool_results(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("first task"),
            tool_call("c1"),
            tool_result("first result", call_id="c1"),
            ChatMessage.from_assistant("first answer"),
            ChatMessage.from_user("second task"),
            ChatMessage.from_assistant("second answer"),
        ]
        spans = _historical_turn_spans(messages=messages, start=1, end=len(messages))
        assert spans == [(1, 5), (5, 7)]
        assert messages[slice(*spans[0])] == messages[1:5]
        assert messages[slice(*spans[1])] == messages[5:7]

    def test_only_returns_turns_within_the_requested_bounds(self):
        messages = [
            ChatMessage.from_user("outside"),
            ChatMessage.from_assistant("outside answer"),
            ChatMessage.from_user("inside"),
            ChatMessage.from_assistant("inside answer"),
            ChatMessage.from_user("current task"),
        ]
        assert _historical_turn_spans(messages=messages, start=2, end=4) == [(2, 4)]

    def test_compaction_note_does_not_start_a_new_turn(self):
        note = ChatMessage.from_user(
            "Earlier messages were removed.", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}
        )
        messages = [
            ChatMessage.from_user("task"),
            ChatMessage.from_assistant("first step"),
            note,
            ChatMessage.from_assistant("second step"),
            ChatMessage.from_user("next task"),
        ]
        assert _historical_turn_spans(messages=messages, start=0, end=len(messages)) == [(0, 4), (4, 5)]


class TestSlidingWindowCompactor:
    def test_replaces_the_middle_with_an_omission_note(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
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
        assert compacted[2].is_from(role=ChatRole.USER)

    def test_places_omission_note_before_current_task_when_only_historical_turns_are_removed(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="old question"),
            ChatMessage.from_assistant(text="old answer " * 100),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current answer"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=20, token_counter=COUNTER)
        assert compacted is not None
        assert compacted[0] == messages[0]
        assert _COMPACTION_META_KEY in compacted[1].meta
        assert compacted[2:] == messages[3:]

    def test_a_roomier_target_keeps_more(self):
        messages = [ChatMessage.from_system(text="rules"), ChatMessage.from_user(text="task")]
        for index in range(4):
            messages.extend([tool_call(f"c{index}"), tool_result(result="x" * 400, call_id=f"c{index}")])
        compactor = SlidingWindowCompactor(omission_note=None)
        tight = compactor.compact(messages=messages, target_tokens=60, token_counter=COUNTER)
        roomy = compactor.compact(messages=messages, target_tokens=350, token_counter=COUNTER)
        assert tight is not None
        assert roomy is not None
        assert len(tight) == 4
        assert len(roomy) == 8

    def test_keeps_complete_recent_user_assistant_turns_that_fit(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="old question"),
            ChatMessage.from_assistant(text="old answer"),
            ChatMessage.from_user(text="recent question"),
            ChatMessage.from_assistant(text="recent answer"),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current step"),
        ]
        # Drops one historical turn
        expected = [messages[0], *messages[3:]]
        target_tokens = COUNTER.count(expected)
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted == expected

    def test_drops_historical_context_and_one_current_task_step_to_reach_target(self):
        system_message = ChatMessage.from_system(text="rules")
        historical_turn = [
            ChatMessage.from_user(text="old question"),
            tool_call("old-call"),
            tool_result(result="old result", call_id="old-call"),
            ChatMessage.from_assistant(text="old final answer"),
        ]
        current_task = [
            ChatMessage.from_user(text="current task"),
            tool_call("current-call-1"),
            tool_result(result="large intermediate result " * 100, call_id="current-call-1"),
            tool_call("current-call-2"),
            tool_result(result="latest result", call_id="current-call-2"),
        ]
        messages = [system_message, *historical_turn, *current_task]
        # The target is small enough that the historical context and one step in the current task must be removed.
        target_tokens = 52
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted == [system_message, current_task[0], *current_task[-2:]]

    def test_returns_none_when_the_conversation_already_fits(self):
        assert (
            SlidingWindowCompactor().compact(messages=long_conversation(), target_tokens=100_000, token_counter=COUNTER)
            is None
        )

    def test_omission_note_can_be_turned_off(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        # System + user + tool_call + tool_result
        assert len(compacted) == 4
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
        compacted = SlidingWindowCompactor(omission_note=note).compact(
            messages=long_conversation(), target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert compacted[2].text == expected

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="empty"),
            pytest.param([ChatMessage.from_system(text="a"), ChatMessage.from_system(text="b")], id="only-system"),
            pytest.param(
                [ChatMessage.from_system(text="rules"), ChatMessage.from_user(text="hi")], id="nothing-outside-window"
            ),
        ],
    )
    def test_returns_none_when_there_is_nothing_to_remove(self, messages):
        assert (
            SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER) is None
        )

    def test_keeps_a_parallel_tool_call_together_with_all_results(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="task"),
            tool_call("old"),
            tool_result(result="old result " * 40, call_id="old"),
            tool_call("a", "b", "c"),
            tool_result(result="first", call_id="a"),
            tool_result(result="second", call_id="b"),
            tool_result(result="third", call_id="c"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert compacted is not None
        assert compacted[-4:] == messages[-4:]
        assert {call.id for call in compacted[-4].tool_calls} == {
            result.tool_call_result.origin.id for result in compacted[-3:]
        }

    @pytest.mark.parametrize(("min_keep_steps", "expected"), [(0, 0), (1, 1), (2, 2), (20, 2)])
    def test_min_keep_steps_wins_over_an_unaffordable_target(self, min_keep_steps, expected):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_steps=min_keep_steps, omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        result = compacted or messages
        assert sum(message.is_from(role=ChatRole.ASSISTANT) for message in result) == expected

    def test_rejects_negative_min_keep_steps(self):
        with pytest.raises(ValueError, match="`min_keep_steps` must be at least 0"):
            SlidingWindowCompactor(min_keep_steps=-1)

    def test_preserves_the_latest_user_task_when_compacting_input_history(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="old question " * 100),
            ChatMessage.from_assistant(text="old answer"),
            ChatMessage.from_user(text="current task"),
        ]
        compacted = SlidingWindowCompactor(min_keep_steps=0, omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted == [messages[0], messages[-1]]

    def test_repeated_compaction_folds_the_previous_note(self):
        compactor = SlidingWindowCompactor()
        first = compactor.compact(messages=long_conversation(), target_tokens=SMALLEST, token_counter=COUNTER)
        assert first is not None
        # Simulate two more turns arriving on top of the already-compacted conversation.
        grown = [*first, tool_call("c3"), tool_result(result="third result", call_id="c3")]
        second = compactor.compact(messages=grown, target_tokens=SMALLEST, token_counter=COUNTER)
        assert second is not None
        # The original task stays anchored and the earlier compaction note is folded into the new one.
        assert count_markers(messages=second) == 1
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
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="task"),
            ChatMessage.from_assistant(text=removable),
            ChatMessage.from_assistant(text="latest step"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert (compacted is not None) is worth_replacing
        # Without a note there is nothing to pay for, so the same cut is always worth making.
        assert SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        ) == [*messages[:2], messages[-1]]

    def test_keeping_no_steps_still_preserves_the_current_task(self):
        messages = long_conversation()
        compacted = SlidingWindowCompactor(min_keep_steps=0, omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted == messages[:2]

    def test_serde_round_trip(self):
        compactor = SlidingWindowCompactor(min_keep_steps=7, omission_note="Dropped {num_removed}.")
        data = compactor.to_dict()
        assert data == {
            "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
            "init_parameters": {"min_keep_steps": 7, "omission_note": "Dropped {num_removed}."},
        }
        restored = SlidingWindowCompactor.from_dict(data=data)
        assert restored.min_keep_steps == 7
        assert restored.omission_note == "Dropped {num_removed}."


class TestSlidingWindowCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        # `SlidingWindowCompactor` does no I/O, so it relies on the protocol's default `compact_async`.
        compactor = SlidingWindowCompactor()
        messages = long_conversation()
        assert await compactor.compact_async(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        ) == compactor.compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
