# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.utils import _estimated_context_tokens, _last_assistant_index
from haystack.tools import tool
from test.hooks.compaction.helpers import FakeCounter, tool_result

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
            pytest.param([ChatMessage.from_user("hi")], -1, id="no-assistant"),
            pytest.param([ChatMessage.from_user("hi"), ChatMessage.from_assistant("yo")], 1, id="assistant-is-last"),
            pytest.param([ChatMessage.from_assistant("yo"), tool_result("r")], 0, id="tool-result-after-assistant"),
            pytest.param(
                [ChatMessage.from_assistant("a"), tool_result("r"), ChatMessage.from_assistant("b"), tool_result("s")],
                2,
                id="takes-the-most-recent",
            ),
        ],
    )
    def test_boundary(self, messages, expected):
        assert _last_assistant_index(messages) == expected


class TestEstimatedContextTokens:
    def test_counts_only_what_the_generator_has_not_seen(self):
        counter = FakeCounter()
        # The reported count covers everything through the assistant reply; only the tool result came after.
        messages = [ChatMessage.from_user("start"), ChatMessage.from_assistant("reply"), tool_result("R" * 400)]
        delta = counter.count(messages[2:])

        assert _estimated_context_tokens(messages, 5000, counter) == 5000 + delta
        assert delta > 0

    def test_equals_the_reported_count_when_nothing_followed(self):
        messages = [ChatMessage.from_user("start"), ChatMessage.from_assistant("reply")]

        assert _estimated_context_tokens(messages, 5000, FakeCounter()) == 5000

    def test_falls_back_to_counting_everything_without_reported_usage(self):
        counter = FakeCounter()
        messages = [ChatMessage.from_user("start"), ChatMessage.from_assistant("reply"), tool_result("R" * 400)]

        assert _estimated_context_tokens(messages, 0, counter, tools=[lookup]) == counter.count(
            messages, tools=[lookup]
        )

    def test_the_written_back_value_does_not_double_count(self):
        # After compacting, the hook writes back the count through the last assistant message. Feeding that straight
        # back in must reproduce the size of the whole conversation, not overshoot it. Counting the two parts separately
        # loses the separator between them, so allow a couple of tokens of slack.
        counter = FakeCounter()
        messages = [ChatMessage.from_user("start"), ChatMessage.from_assistant("reply"), tool_result("R" * 400)]

        written = counter.count(messages[: _last_assistant_index(messages) + 1])

        assert _estimated_context_tokens(messages, written, counter) == pytest.approx(counter.count(messages), abs=2)
