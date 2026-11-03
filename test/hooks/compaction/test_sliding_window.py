# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.hooks.compaction import SlidingWindowCompactor
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE, _is_compaction_note
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from test.hooks.compaction.helpers import (
    FakeCounter,
    count_markers,
    fresh_conversation_with_two_steps,
    tool_call,
    tool_result,
)

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

# A target of one token forces the window down to `min_keep_steps`, isolating the structural rules from sizing.
SMALLEST = 1
# A fixed, obvious rate, so the tests that do exercise sizing can reason in round numbers.
COUNTER = FakeCounter()


def summary(text: str = "constraint: never delete production data") -> ChatMessage:
    """A user-role summary exactly as `SummarizationCompactor` leaves it: a user message carrying its strategy."""
    return ChatMessage.from_user(
        text=f"<conversation_summary>\n{text}\n</conversation_summary>",
        meta={_COMPACTION_META_KEY: {"strategy": "summarization", "summarized_messages": 6}},
    )


class TestIsCompactionNote:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            pytest.param(
                ChatMessage.from_user(
                    text="Earlier messages were removed.", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}
                ),
                True,
                id="note-this-strategy-left",
            ),
            # A pruned result carries the same meta key but is still part of the conversation, so it is not a note.
            pytest.param(
                ChatMessage.from_tool(
                    tool_result="[Tool result removed to free up context.]",
                    origin=ToolCall(tool_name="search", arguments={}, id="c1"),
                    meta={_COMPACTION_META_KEY: {"strategy": "tool_result_pruning", "original_tokens": 180}},
                ),
                False,
                id="tool-result-another-strategy-pruned",
            ),
            # Another strategy's note is not this one's to fold away, so it is left where it is.
            pytest.param(
                ChatMessage.from_user(
                    text="A summary of what came before.", meta={_COMPACTION_META_KEY: {"strategy": "summarization"}}
                ),
                False,
                id="note-another-strategy-left",
            ),
            pytest.param(
                ChatMessage.from_system(text="rules", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}),
                False,
                id="system-message",
            ),
            pytest.param(
                ChatMessage.from_user(text="odd", meta={_COMPACTION_META_KEY: "sliding_window"}),
                False,
                id="marker-that-is-not-a-dict",
            ),
        ],
    )
    def test_only_matches_sliding_window_omission_message(self, message, expected):
        assert _is_compaction_note(message=message) is expected


class TestSlidingWindowCompactor:
    def test_replaces_all_historical_turns(self):
        messages = [
            # System
            ChatMessage.from_system(text="rules"),
            # Historical turn
            ChatMessage.from_user(text="old question " * 100),
            ChatMessage.from_assistant(text="old answer"),
            # Current task
            ChatMessage.from_user(text="current task"),
        ]
        compacted = SlidingWindowCompactor(min_keep_steps=0, omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted == [messages[0], messages[-1]]

    def test_replaces_oldest_agent_step(self):
        messages = fresh_conversation_with_two_steps()
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert compacted is not None
        # Only the oldest agent step is removed which is now where the omission note is.
        assert compacted == [*messages[:2], compacted[2], *messages[4:]]

        # Test omission note
        assert compacted[2].meta[_COMPACTION_META_KEY] == {
            "strategy": "sliding_window",
            "removed_messages": 2,
            "kept_messages": 4,
        }
        assert compacted[2].text == _DEFAULT_OMISSION_NOTE.replace("{num_removed}", "2")
        assert compacted[2].is_from(role=ChatRole.USER)

    def test_replaces_oldest_historical_turn(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="old question"),
            ChatMessage.from_assistant(text="old answer " * 100),
            ChatMessage.from_user(text="recent question"),
            ChatMessage.from_assistant(text="recent answer"),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current answer"),
        ]
        # Enough for the instructions, the recent turn, and the task with its step, but not the padded oldest turn.
        target_tokens = 30
        compacted = SlidingWindowCompactor().compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted is not None
        # The oldest historical turn is removed and replaced with an omission note.
        assert compacted == [messages[0], compacted[1], *messages[3:]]
        assert _COMPACTION_META_KEY in compacted[1].meta

    def test_drops_all_historical_turns_and_oldest_agent_step(self):
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

    def test_replaces_earlier_note_and_oldest_historical_turn(self):
        messages = [
            # System
            ChatMessage.from_system(text="rules"),
            # The note an earlier compaction left, which sits at the top of the historical turns.
            ChatMessage.from_user(
                text="Earlier messages were removed.", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}
            ),
            # Historical
            ChatMessage.from_user(text="old question " * 200),
            ChatMessage.from_assistant(text="old answer"),
            ChatMessage.from_user(text="recent question"),
            ChatMessage.from_assistant(text="recent answer"),
            # Current Task
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current step"),
        ]
        # Enough for the instructions, the recent turn, and the task with its step, but not the padded oldest turn.
        target_tokens = 30
        compacted = SlidingWindowCompactor().compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted is not None
        # The earlier note goes with the turn it stood in front of, and the new note takes its place.
        assert compacted == [messages[0], compacted[1], *messages[4:]]
        assert count_markers(messages=compacted) == 1
        # The earlier note is counted among the removed, alongside the two messages of the oldest turn.
        assert compacted[1].meta[_COMPACTION_META_KEY]["removed_messages"] == 3

    def test_replaces_earlier_note_and_oldest_agent_step(self):
        messages = [
            # System
            ChatMessage.from_system(text="rules"),
            # Current Task
            ChatMessage.from_user(text="current task"),
            # The note an earlier compaction left, which sits right after the task when its own steps were trimmed.
            ChatMessage.from_user(
                text="Earlier messages were removed.", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}
            ),
            tool_call("c1"),
            tool_result(result="first result", call_id="c1"),
            tool_call("c2"),
            tool_result(result="second result", call_id="c2"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert compacted is not None
        # The earlier note goes with the step it stood in front of, and the new note takes its place.
        assert compacted == [*messages[:2], compacted[2], *messages[5:]]
        assert count_markers(messages=compacted) == 1
        # The earlier note is counted among the removed, alongside the two messages of the oldest step.
        assert compacted[2].meta[_COMPACTION_META_KEY]["removed_messages"] == 3

    def test_keeps_a_pruned_tool_result_inside_a_kept_turn(self):
        messages = [
            # System
            ChatMessage.from_system(text="rules"),
            # Historical, dropped to make room
            ChatMessage.from_user(text="ancient question " * 200),
            ChatMessage.from_assistant(text="ancient answer"),
            # Historical, kept
            ChatMessage.from_user(text="old question"),
            tool_call("old"),
            # A result `ToolResultPruningCompactor` already pruned, which carries the same meta key as an omission note
            # but is part of the conversation rather than standing in for removed history.
            ChatMessage.from_tool(
                tool_result="[Tool result removed to free up context.]",
                origin=ToolCall(tool_name="search", arguments={}, id="old"),
                meta={_COMPACTION_META_KEY: {"strategy": "tool_result_pruning", "original_tokens": 180}},
            ),
            # Current Task
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current step"),
        ]
        # Enough for the instructions, the kept turn, and the task with its step, but not the padded oldest turn.
        target_tokens = 45
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=target_tokens, token_counter=COUNTER
        )
        assert compacted is not None
        # The pruned result is not an omission note, so it stays with the turn and its tool call keeps its answer.
        assert compacted == [messages[0], *messages[3:]]

    def test_omission_note_can_be_turned_off(self):
        messages = fresh_conversation_with_two_steps()
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
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
            messages=fresh_conversation_with_two_steps(), target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert compacted[2].text == expected

    def test_returns_none_when_only_an_earlier_note_would_be_removed(self):
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(
                text=_DEFAULT_OMISSION_NOTE.replace("{num_removed}", "12"),
                meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}},
            ),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current step"),
        ]
        # Enough for everything but the earlier note, leaving that note as the only thing compaction could remove.
        target_tokens = 16
        # Swapping one note for another frees nothing, so compaction must decline rather than run again every step.
        assert (
            SlidingWindowCompactor().compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)
            is None
        )

    @pytest.mark.parametrize(
        ("messages", "target_tokens"),
        [
            # The conversation is already under the target, so there is nothing to do.
            pytest.param(fresh_conversation_with_two_steps(), 100_000, id="conversation-already-fits"),
            # Over the target, but everything that is left is protected, so there is nothing the compactor may remove.
            pytest.param(
                [ChatMessage.from_system(text="a"), ChatMessage.from_system(text="b")], SMALLEST, id="only-system"
            ),
            pytest.param(
                [ChatMessage.from_system(text="rules"), ChatMessage.from_user(text="hi")],
                SMALLEST,
                id="only-system-and-task",
            ),
        ],
    )
    def test_returns_none_when_there_is_nothing_to_remove(self, messages, target_tokens):
        assert (
            SlidingWindowCompactor().compact(messages=messages, target_tokens=target_tokens, token_counter=COUNTER)
            is None
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
        tool_call_results = [result.tool_call_result for result in compacted[-3:]]
        assert all(tool_call_result is not None for tool_call_result in tool_call_results)
        assert {call.id for call in compacted[-4].tool_calls} == {
            tool_call_result.origin.id for tool_call_result in tool_call_results if tool_call_result is not None
        }

    @pytest.mark.parametrize(("min_keep_steps", "expected"), [(0, 0), (1, 1), (2, 2), (20, 2)])
    def test_min_keep_steps_wins_over_an_unaffordable_target(self, min_keep_steps, expected):
        messages = fresh_conversation_with_two_steps()
        compacted = SlidingWindowCompactor(min_keep_steps=min_keep_steps, omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        result = compacted or messages
        assert sum(message.is_from(role=ChatRole.ASSISTANT) for message in result) == expected

    def test_rejects_negative_min_keep_steps(self):
        with pytest.raises(ValueError, match="`min_keep_steps` must be at least 0"):
            SlidingWindowCompactor(min_keep_steps=-1)

    def test_keeping_no_steps_still_preserves_the_current_task(self):
        messages = fresh_conversation_with_two_steps()
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
        assert isinstance(restored, SlidingWindowCompactor)
        assert restored.min_keep_steps == 7
        assert restored.omission_note == "Dropped {num_removed}."


class TestKeepsAnotherStrategysSummary:
    # These cover the combination documented for compactors: a summarization compactor runs first and leaves a user
    # summary; a later sliding-window pass must treat that summary as retained memory, not as a message it may drop.

    def test_keeps_a_historical_summary_when_all_turns_are_removed(self):
        historical_summary = summary("history constraint: the user is in the EU, keep GDPR wording")
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="ancient question " * 200),
            ChatMessage.from_assistant(text="ancient answer"),
            historical_summary,
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current answer"),
        ]
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # The historical summary is retained ahead of the task even though the padded ancient turn does not fit.
        assert historical_summary in compacted
        assert compacted == [messages[0], historical_summary, messages[4], messages[5]]

    def test_places_the_note_before_a_retained_historical_summary(self):
        historical_summary = summary("history constraint: the user is in the EU")
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="ancient question " * 200),
            ChatMessage.from_assistant(text="ancient answer"),
            historical_summary,
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current answer"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert compacted is not None
        # The note stands in for the dropped ancient turn, with the surviving summary after it in conversation order.
        assert compacted[1].meta[_COMPACTION_META_KEY]["strategy"] == "sliding_window"
        assert compacted == [messages[0], compacted[1], historical_summary, messages[4], messages[5]]

    def test_keeps_a_current_task_summary_when_oldest_step_is_removed(self):
        current_summary = summary("current-task constraint: never delete production data; ticket ABC-123")
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="current task"),
            current_summary,
            tool_call("c1"),
            tool_result(result="large first result " * 60, call_id="c1"),
            tool_call("c2"),
            tool_result(result="newest result", call_id="c2"),
        ]
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        # The summary is not part of the oldest step being dropped, so it survives with the retained newest step.
        assert current_summary in compacted
        assert compacted == [messages[0], messages[1], current_summary, *messages[5:]]

    def test_places_the_note_between_the_task_and_a_retained_current_task_summary(self):
        current_summary = summary("current-task constraint: never delete production data")
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="current task"),
            current_summary,
            tool_call("c1"),
            tool_result(result="large first result " * 60, call_id="c1"),
            tool_call("c2"),
            tool_result(result="newest result", call_id="c2"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
        assert compacted is not None
        # The omission note follows the task anchor; the summary and the retained newest step come after it.
        assert compacted[2].meta[_COMPACTION_META_KEY]["strategy"] == "sliding_window"
        assert compacted == [messages[0], messages[1], compacted[2], current_summary, *messages[5:]]

    def test_keeps_a_summary_between_two_historical_turns(self):
        historical_summary = summary("constraint carried out of the first turn")
        messages = [
            ChatMessage.from_system(text="rules"),
            ChatMessage.from_user(text="ancient question " * 40),
            ChatMessage.from_assistant(text="ancient answer"),
            historical_summary,
            ChatMessage.from_user(text="recent question"),
            ChatMessage.from_assistant(text="recent answer"),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current answer"),
        ]
        # The recent turn, task and summary fit together, the padded ancient turn does not; that turn is removed and
        # the summary between the two turns must be retained.
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=50, token_counter=COUNTER
        )
        assert compacted is not None
        assert historical_summary in compacted
        assert compacted == [messages[0], historical_summary, *messages[4:]]

    def test_keeps_a_summary_with_no_task_anchor(self):
        # With no real user message to anchor on, the summary still belongs to protected retained context and the
        # oldest current step is the one removed.
        current_summary = summary("constraint established without a task anchor")
        messages = [
            ChatMessage.from_system(text="rules"),
            current_summary,
            tool_call("c1"),
            tool_result(result="large first result " * 60, call_id="c1"),
            tool_call("c2"),
            tool_result(result="newest result", call_id="c2"),
        ]
        compacted = SlidingWindowCompactor(omission_note=None).compact(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        )
        assert compacted is not None
        assert current_summary in compacted
        assert compacted == [messages[0], current_summary, *messages[4:]]

    def test_does_not_treat_a_sliding_window_note_as_a_protected_summary(self):
        # Our own omission notes remain foldable: a later pass must still be able to swap them for a fresh note.
        earlier_note = ChatMessage.from_user(
            text=_DEFAULT_OMISSION_NOTE.replace("{num_removed}", "4"),
            meta={_COMPACTION_META_KEY: {"strategy": "sliding_window", "removed_messages": 4, "kept_messages": 2}},
        )
        messages = [
            ChatMessage.from_system(text="rules"),
            earlier_note,
            ChatMessage.from_user(text="old question " * 200),
            ChatMessage.from_assistant(text="old answer"),
            ChatMessage.from_user(text="current task"),
            ChatMessage.from_assistant(text="current step"),
        ]
        compacted = SlidingWindowCompactor().compact(messages=messages, target_tokens=30, token_counter=COUNTER)
        assert compacted is not None
        # The earlier sliding-window note is folded into the new note, while a summarization summary would be kept.
        assert earlier_note not in compacted
        assert compacted[1].meta[_COMPACTION_META_KEY]["strategy"] == "sliding_window"
        assert count_markers(messages=compacted) == 1


class TestSlidingWindowCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        # `SlidingWindowCompactor` does no I/O, so it relies on the protocol's default `compact_async`.
        compactor = SlidingWindowCompactor()
        messages = fresh_conversation_with_two_steps()
        assert await compactor.compact_async(
            messages=messages, target_tokens=SMALLEST, token_counter=COUNTER
        ) == compactor.compact(messages=messages, target_tokens=SMALLEST, token_counter=COUNTER)
