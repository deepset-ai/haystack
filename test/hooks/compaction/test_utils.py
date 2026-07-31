# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _compaction_split,
    _estimated_context_tokens,
    _last_assistant_index,
)
from test.hooks.compaction.helpers import FakeCounter, tool_call, tool_result

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

COMPACTED_META = {_COMPACTION_META_KEY: {"strategy": "sliding_window"}}

# Each message is padded so a `FakeCounter` at 4 chars per token gives it a round, predictable cost.
TWENTY_FIVE_TOKENS = "x" * 91  # "[user] " + 91 chars -> 98 // 4 = 24, close enough to reason in tens


def _sized(chars: int) -> ChatMessage:
    """A user message whose rendered form is roughly `chars` characters."""
    return ChatMessage.from_user("x" * chars)


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

        assert _estimated_context_tokens(messages, 0, counter) == counter.count(messages)

    def test_the_written_back_value_does_not_double_count(self):
        # After compacting, the hook writes back the count through the last assistant message. Feeding that straight
        # back in must reproduce the size of the whole conversation, not overshoot it. Counting the two parts separately
        # loses the separator between them, so allow a couple of tokens of slack.
        counter = FakeCounter()
        messages = [ChatMessage.from_user("start"), ChatMessage.from_assistant("reply"), tool_result("R" * 400)]

        written = counter.count(messages[: _last_assistant_index(messages) + 1])

        assert _estimated_context_tokens(messages, written, counter) == pytest.approx(counter.count(messages), abs=2)


class TestCompactionSplit:
    def test_window_grows_to_fill_the_target(self):
        # Ten messages of ~25 tokens each; a 60-token target should keep roughly the last two.
        messages = [_sized(100) for _ in range(10)]
        kept_prefix, _, kept_window = _compaction_split(
            messages, target_tokens=60, token_counter=FakeCounter(), min_keep_messages=1
        )

        assert kept_prefix == []
        assert 2 <= len(kept_window) <= 3, f"kept {len(kept_window)} messages for a 60-token target"

    def test_a_bigger_target_keeps_more(self):
        messages = [_sized(100) for _ in range(10)]
        *_, tight = _compaction_split(messages, target_tokens=60, token_counter=FakeCounter(), min_keep_messages=1)
        *_, roomy = _compaction_split(messages, target_tokens=200, token_counter=FakeCounter(), min_keep_messages=1)

        assert len(roomy) > len(tight)

    def test_nothing_is_removable_when_it_already_fits(self):
        messages = [_sized(20), _sized(20)]
        _, removable, _ = _compaction_split(
            messages, target_tokens=100_000, token_counter=FakeCounter(), min_keep_messages=1
        )

        assert removable == []

    def test_min_keep_messages_wins_over_an_unaffordable_target(self):
        messages = [_sized(400) for _ in range(6)]
        *_, kept_window = _compaction_split(messages, target_tokens=1, token_counter=FakeCounter(), min_keep_messages=3)

        assert len(kept_window) == 3

    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param([], (0, 0), id="empty"),
            # The Agent's standing instructions are never removable.
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], (2, 2), id="only-system"),
            pytest.param(
                [ChatMessage.from_user("hi"), ChatMessage.from_system("late rules")],
                (0, 1),
                id="system-not-leading-does-not-extend-the-protected-run",
            ),
            # A note an earlier compaction produced is removable, so the next one replaces it.
            pytest.param(
                [
                    ChatMessage.from_system("rules"),
                    ChatMessage.from_system("earlier note", meta=COMPACTED_META),
                    ChatMessage.from_user("hi"),
                ],
                (1, 2),
                id="previous-compaction-is-removable",
            ),
            # The window may not start on a tool result whose call is about to be removed.
            pytest.param(
                [ChatMessage.from_user("hi"), tool_call("c1"), tool_result("found", call_id="c1")],
                (0, 1),
                id="window-grows-past-tool-result",
            ),
            pytest.param(
                [
                    ChatMessage.from_user("hi"),
                    tool_call("c0", "c1", "c2"),
                    tool_result("a", call_id="c0"),
                    tool_result("b", call_id="c1"),
                    tool_result("c", call_id="c2"),
                ],
                (0, 1),
                id="window-grows-past-parallel-batch",
            ),
            pytest.param(
                [ChatMessage.from_system("rules"), tool_result("orphan")],
                (1, 1),
                id="window-never-grows-into-the-system-block",
            ),
        ],
    )
    def test_structural_rules(self, messages, expected):
        # A target of 1 token forces the window as small as the structural rules allow, isolating them from sizing.
        start, end = expected

        split = _compaction_split(messages, target_tokens=1, token_counter=FakeCounter(), min_keep_messages=1)

        assert split == (messages[:start], messages[start:end], messages[end:])
