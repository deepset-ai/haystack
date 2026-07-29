# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _compaction_bounds
from test.hooks.compaction.helpers import tool_call, tool_result

COMPACTED_META = {_COMPACTION_META_KEY: {"strategy": "sliding_window"}}


class TestCompactionBounds:
    @pytest.mark.parametrize(
        ("messages", "keep_last_n", "expected"),
        [
            pytest.param([], 5, (0, 0), id="empty"),
            pytest.param([ChatMessage.from_user(f"m{i}") for i in range(6)], 2, (0, 4), id="keeps-requested-count"),
            pytest.param([ChatMessage.from_user("hi")], 1, (0, 0), id="nothing-to-remove"),
            # The Agent's standing instructions are never removable.
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], 1, (2, 2), id="only-system"),
            pytest.param(
                [ChatMessage.from_system("a"), ChatMessage.from_system("b"), ChatMessage.from_user("hi")],
                1,
                (2, 2),
                id="leading-system-block",
            ),
            pytest.param(
                [ChatMessage.from_user("hi"), ChatMessage.from_system("late rules")],
                1,
                (0, 1),
                id="system-not-leading-does-not-extend-the-protected-run",
            ),
            # Only the leading run of system messages is protected. One injected mid-conversation, such as a per-step
            # nudge from a hook, is removable like anything else - keeping every one would pile them up.
            pytest.param(
                [
                    ChatMessage.from_system("rules"),
                    ChatMessage.from_user("a"),
                    ChatMessage.from_system("nudge"),
                    ChatMessage.from_user("b"),
                    ChatMessage.from_user("c"),
                ],
                2,
                (1, 3),
                id="mid-conversation-system-is-removable",
            ),
            # A note or summary an earlier compaction produced is removable, so the next one replaces it.
            pytest.param(
                [
                    ChatMessage.from_system("rules"),
                    ChatMessage.from_system("earlier note", meta=COMPACTED_META),
                    ChatMessage.from_user("hi"),
                ],
                1,
                (1, 2),
                id="previous-compaction-is-removable",
            ),
            # The tail may not start on a tool result whose call is about to be removed, so it grows to include the
            # assistant message holding the call.
            pytest.param(
                [ChatMessage.from_user("hi"), tool_call("c1"), tool_result("found", call_id="c1")],
                1,
                (0, 1),
                id="tail-grows-past-tool-result",
            ),
            pytest.param(
                [
                    ChatMessage.from_user("hi"),
                    tool_call("c0", "c1", "c2"),
                    tool_result("a", call_id="c0"),
                    tool_result("b", call_id="c1"),
                    tool_result("c", call_id="c2"),
                ],
                1,
                (0, 1),
                id="tail-grows-past-parallel-batch",
            ),
            pytest.param(
                [ChatMessage.from_system("rules"), tool_result("orphan")],
                1,
                (1, 1),
                id="tail-never-grows-into-the-system-block",
            ),
            pytest.param(
                [ChatMessage.from_system("rules"), ChatMessage.from_user("hi")],
                50,
                (1, 1),
                id="keeps-more-than-there-is",
            ),
        ],
    )
    def test_bounds(self, messages, keep_last_n, expected):
        assert _compaction_bounds(messages, keep_last_n) == expected
