# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.utils import _agent_step_spans, _estimated_context_tokens, _last_assistant_index
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
    def test_groups_assistant_messages_with_all_immediately_following_tool_results(self):
        messages = [
            ChatMessage.from_user("task"),
            tool_call("parallel-1", "parallel-2"),
            tool_result("first", call_id="parallel-1"),
            tool_result("second", call_id="parallel-2"),
            ChatMessage.from_user("next task"),
            ChatMessage.from_assistant("plain answer"),
            tool_call("later"),
            tool_result("later result", call_id="later"),
        ]
        assert _agent_step_spans(messages=messages, start=0) == [(1, 4), (5, 6), (6, 8)]

    def test_starts_at_the_requested_message(self):
        messages = [tool_call("old"), tool_result("old result", call_id="old"), tool_call("current")]
        assert _agent_step_spans(messages=messages, start=2) == [(2, 3)]


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
