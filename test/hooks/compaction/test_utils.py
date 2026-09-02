# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _agent_step_spans,
    _current_agent_step_groups,
    _estimated_context_tokens,
    _historical_turn_groups,
    _historical_turn_spans,
    _is_compaction_message,
    _last_assistant_index,
)
from haystack.tools import tool
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


@tool
def lookup(query: str) -> str:
    """Look up information relevant to a query."""
    return query


class TestLastAssistantIndex:
    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param([], -1, id="empty"),
            pytest.param([ChatMessage.from_user(text="hi")], -1, id="no-assistant"),
            pytest.param(
                [ChatMessage.from_user(text="hi"), ChatMessage.from_assistant(text="yo")], 1, id="assistant-is-last"
            ),
            pytest.param(
                [ChatMessage.from_assistant(text="yo"), tool_result(result="r")], 0, id="tool-result-after-assistant"
            ),
            pytest.param(
                [
                    ChatMessage.from_assistant(text="a"),
                    tool_result(result="r"),
                    ChatMessage.from_assistant(text="b"),
                    tool_result(result="s"),
                ],
                2,
                id="takes-the-most-recent",
            ),
        ],
    )
    def test_boundary(self, messages, expected):
        assert _last_assistant_index(messages=messages) == expected


class TestAgentStepSpans:
    def test_single_assistant_message_is_one_step(self):
        messages = [ChatMessage.from_user("task"), ChatMessage.from_assistant("plain answer")]
        # A text-only assistant turn has no tool results to extend its span, so the step contains one message.
        assert _agent_step_spans(messages=messages, start=0) == [(1, 2)]

    def test_complex_agent_steps(self):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("parallel-1", "parallel-2"),
            tool_result("first", call_id="parallel-1"),
            tool_result("second", call_id="parallel-2"),
            ChatMessage.from_user("next task"),
            ChatMessage.from_assistant("plain answer"),
            ChatMessage.from_user("follow-up task"),
            tool_call("later"),
            tool_result("later result", call_id="later"),
        ]
        assert _agent_step_spans(messages=messages, start=0) == [(1, 4), (5, 6), (7, 9)]

    def test_starts_at_the_requested_message(self):
        messages = [tool_call("old"), tool_result("old result", call_id="old"), tool_call("current")]
        assert _agent_step_spans(messages=messages, start=2) == [(2, 3)]


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
        messages = [
            # Historical turns
            ChatMessage.from_user(
                "Earlier messages were removed.", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}}
            ),
            ChatMessage.from_user("task"),
            ChatMessage.from_assistant("first step"),
            ChatMessage.from_user("next task"),
            ChatMessage.from_assistant("second step"),
        ]
        # The note is skipped which is why the first span starts at 1
        assert _historical_turn_spans(messages=messages, start=0, end=len(messages)) == [(1, 3), (3, 5)]


class TestIsCompactionMessage:
    @pytest.mark.parametrize(
        ("strategy", "role", "expected"),
        [
            pytest.param("sliding_window", None, True, id="matching-strategy-any-role"),
            pytest.param("summarization", None, False, id="another-strategy"),
            pytest.param("sliding_window", ChatRole.USER, True, id="matching-strategy-and-role"),
            pytest.param("sliding_window", ChatRole.SYSTEM, False, id="matching-strategy-wrong-role"),
        ],
    )
    def test_strategy_and_role(self, strategy, role, expected):
        note = ChatMessage.from_user(text="removed", meta={_COMPACTION_META_KEY: {"strategy": "sliding_window"}})
        assert _is_compaction_message(message=note, strategy=strategy, role=role) is expected

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param(ChatMessage.from_user(text="hi"), id="no-marker"),
            # A marker that is not a dict cannot carry a strategy, so it matches nothing.
            pytest.param(
                ChatMessage.from_user(text="odd", meta={_COMPACTION_META_KEY: "sliding_window"}),
                id="marker-that-is-not-a-dict",
            ),
        ],
    )
    def test_unusable_marker(self, message):
        assert _is_compaction_message(message=message, strategy="sliding_window") is False


class TestHistoricalTurnGroups:
    def test_basic(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("old question"),
            ChatMessage.from_assistant("old answer"),
            ChatMessage.from_user("current task"),
        ]
        assert _historical_turn_groups(messages=messages, system_end=1, task_index=3) == [[1, 2]]

    def test_missing_task_anchor(self):
        # With no user message to anchor on, everything after the system block belongs to the current task instead.
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_assistant("step")]
        assert _historical_turn_groups(messages=messages, system_end=1, task_index=None) == []


class TestCurrentAgentStepGroups:
    def test_basic(self):
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("current task"),
            tool_call("c1"),
            tool_result("result", call_id="c1"),
            ChatMessage.from_assistant("answer"),
        ]
        assert _current_agent_step_groups(messages=messages, system_end=1, task_index=1) == [[2, 3], [4]]

    def test_missing_task_anchor(self):
        messages = [ChatMessage.from_system("rules"), ChatMessage.from_assistant("step")]
        assert _current_agent_step_groups(messages=messages, system_end=1, task_index=None) == [[1]]


class TestEstimatedContextTokens:
    def test_counts_only_what_the_generator_has_not_seen(self):
        counter = FakeCounter()
        # The reported count covers everything through the assistant reply; only the tool result came after.
        messages = [
            ChatMessage.from_user(text="start"),
            ChatMessage.from_assistant(text="reply"),
            tool_result(result="R" * 400),
        ]
        delta = counter.count(messages=messages[2:])
        assert _estimated_context_tokens(messages=messages, context_tokens=5000, token_counter=counter) == 5000 + delta
        assert delta > 0

    def test_equals_the_reported_count_when_nothing_followed(self):
        messages = [ChatMessage.from_user(text="start"), ChatMessage.from_assistant(text="reply")]
        assert _estimated_context_tokens(messages=messages, context_tokens=5000, token_counter=FakeCounter()) == 5000

    def test_falls_back_to_counting_everything_without_reported_usage(self):
        counter = FakeCounter()
        messages = [
            ChatMessage.from_user(text="start"),
            ChatMessage.from_assistant(text="reply"),
            tool_result(result="R" * 400),
        ]
        assert _estimated_context_tokens(
            messages=messages, context_tokens=0, token_counter=counter, tools=[lookup]
        ) == counter.count(messages=messages, tools=[lookup])

    def test_the_written_back_value_does_not_double_count(self):
        # After compacting, the hook writes back the count through the last assistant message. Feeding that straight
        # back in must reproduce the size of the whole conversation, not overshoot it. Counting the two parts separately
        # loses the separator between them, so allow a couple of tokens of slack.
        counter = FakeCounter()
        messages = [
            ChatMessage.from_user(text="start"),
            ChatMessage.from_assistant(text="reply"),
            tool_result(result="R" * 400),
        ]
        written = counter.count(messages=messages[: _last_assistant_index(messages=messages) + 1])
        assert _estimated_context_tokens(
            messages=messages, context_tokens=written, token_counter=counter
        ) == pytest.approx(counter.count(messages=messages), abs=2)
