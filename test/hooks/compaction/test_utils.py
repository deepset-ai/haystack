# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _agent_step_spans,
    _current_agent_step_groups,
    _historical_turn_groups,
    _historical_turn_spans,
    _is_compaction_message,
    _last_assistant_index,
    _replace_tool_results_until_target,
)
from test.hooks.compaction.helpers import FakeCounter, conversation, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


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


COUNTER = FakeCounter(chars_per_token=1)


def _shorten(message: ChatMessage, index: int) -> tuple[ChatMessage, int]:
    """Replace a tool result with a short marker naming its position."""
    assert message.tool_call_result is not None
    shortened = ChatMessage.from_tool(tool_result=f"replaced {index}", origin=message.tool_call_result.origin)
    return shortened, COUNTER.count(messages=[message]) - COUNTER.count(messages=[shortened])


def _replaced_positions(messages: list[ChatMessage]) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if message.tool_call_result is not None and message.tool_call_result.result == f"replaced {index}"
    ]


class TestReplaceToolResultsUntilTarget:
    def test_stops_after_reaching_target(self):
        messages = conversation("a" * 400, "b" * 400, "newest")
        compacted = _replace_tool_results_until_target(
            messages=messages,
            target_tokens=COUNTER.count(messages=messages) - 1,
            token_counter=COUNTER,
            min_keep_steps=1,
            replace=_shorten,
        )

        assert compacted is not None
        assert _replaced_positions(compacted) == [2]
        # The caller-owned input list is unchanged.
        assert _replaced_positions(messages) == []

    @pytest.mark.parametrize(
        ("min_keep_steps", "expected"),
        [pytest.param(1, [2, 4, 5], id="keeps_newest_step"), pytest.param(2, [2], id="keeps_parallel_step_together")],
    )
    def test_keeps_min_keep_steps(self, min_keep_steps, expected):
        messages = [
            *conversation("old" * 200),
            tool_call("parallel-1", "parallel-2"),
            tool_result("first" * 200, call_id="parallel-1"),
            tool_result("second" * 200, call_id="parallel-2"),
            tool_call("newest"),
            tool_result("newest" * 200, call_id="newest"),
        ]
        compacted = _replace_tool_results_until_target(
            messages=messages, target_tokens=1, token_counter=COUNTER, min_keep_steps=min_keep_steps, replace=_shorten
        )

        assert compacted is not None
        assert _replaced_positions(compacted) == expected

    @pytest.mark.parametrize(
        ("messages", "target_tokens", "replace"),
        [
            pytest.param(conversation("a" * 400, "newest"), 10_000, _shorten, id="conversation_already_fits"),
            pytest.param(conversation("only result"), 1, _shorten, id="every_step_is_protected"),
            pytest.param(
                conversation("a" * 400, "newest"), 1, lambda message, index: None, id="nothing_is_replaceable"
            ),
        ],
    )
    def test_returns_none(self, messages, target_tokens, replace):
        assert (
            _replace_tool_results_until_target(
                messages=messages, target_tokens=target_tokens, token_counter=COUNTER, min_keep_steps=1, replace=replace
            )
            is None
        )
