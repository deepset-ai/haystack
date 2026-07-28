# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.compaction import SlidingWindowCompactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY

SCHEMA = {"messages": {"type": list[ChatMessage]}, "step_count": {"type": int}}


def _tool_result(name: str, result: str, *, call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_tool(tool_result=result, origin=ToolCall(tool_name=name, arguments={}, id=call_id))


def _tool_call(name: str, *, call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name=name, arguments={}, id=call_id)])


def _state(messages: list[ChatMessage], *, step_count: int = 3) -> State:
    return State(schema=SCHEMA, data={"messages": messages, "step_count": step_count})


def _long_conversation() -> list[ChatMessage]:
    return [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("start"),
        _tool_call("search", call_id="c1"),
        _tool_result("search", "first", call_id="c1"),
        _tool_call("search", call_id="c2"),
        _tool_result("search", "second", call_id="c2"),
    ]


def _marker(message: ChatMessage) -> dict:
    return message.meta[_COMPACTION_META_KEY]


class TestSlidingWindowCompactor:
    def test_replaces_the_middle_with_an_omission_note(self):
        messages = _long_conversation()
        compacted = SlidingWindowCompactor(keep_last_n_messages=2).compact(_state(messages))

        assert compacted is not None
        # The system prefix survives, the note stands in for what was dropped, and the window is kept verbatim.
        assert compacted[0] is messages[0]
        assert compacted[2:] == messages[4:]
        assert _marker(compacted[1]) == {
            "step": 3,
            "strategy": "sliding_window",
            "removed_messages": 3,
            "kept_messages": 2,
        }
        assert "3 earlier messages" in compacted[1].text

    def test_omission_note_can_be_turned_off(self):
        messages = _long_conversation()
        compacted = SlidingWindowCompactor(keep_last_n_messages=2, omission_note=False).compact(_state(messages))

        assert compacted == [messages[0], *messages[4:]]

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([], id="empty"),
            pytest.param([ChatMessage.from_system("a"), ChatMessage.from_system("b")], id="only-system"),
            pytest.param([ChatMessage.from_system("rules"), ChatMessage.from_user("hi")], id="already-fits"),
        ],
    )
    def test_returns_none_when_there_is_nothing_to_remove(self, messages):
        assert SlidingWindowCompactor(keep_last_n_messages=20).compact(_state(messages)) is None

    def test_keeps_a_tool_call_together_with_its_results(self):
        # Keeping exactly one message would start the window at a tool result whose call had been dropped, which
        # chat-completion APIs reject. The window grows instead.
        messages = _long_conversation()
        compacted = SlidingWindowCompactor(keep_last_n_messages=1).compact(_state(messages))

        assert compacted is not None
        assert compacted[-2:] == messages[4:]
        assert compacted[-2].tool_calls[0].id == compacted[-1].tool_call_result.origin.id

    @pytest.mark.parametrize("keep_last_n_messages", [1, 2, 3, 20])
    def test_never_drops_the_final_message(self, keep_last_n_messages):
        # The Agent may be mid-step with a pending tool call whose result is appended right after compaction returns.
        messages = _long_conversation()
        compacted = SlidingWindowCompactor(keep_last_n_messages=keep_last_n_messages).compact(_state(messages))

        assert (compacted or messages)[-1] is messages[-1]

    def test_repeated_compaction_folds_the_previous_note(self):
        compactor = SlidingWindowCompactor(keep_last_n_messages=2)
        first = compactor.compact(_state(_long_conversation()))
        assert first is not None

        # Simulate two more turns arriving on top of the already-compacted conversation.
        grown = [*first, _tool_call("search", call_id="c3"), _tool_result("search", "third", call_id="c3")]
        second = compactor.compact(_state(grown))

        assert second is not None
        # Exactly one note: the earlier one is inside the block that was dropped, not carried along beside the new one.
        assert sum(_COMPACTION_META_KEY in message.meta for message in second) == 1
        assert second[0].text == "rules"

    def test_returns_none_when_the_note_would_cost_what_it_saves(self):
        # Dropping one message to add a note in its place gains nothing, and repeating it every step would just swap
        # one note for the next.
        # Four messages with a one-message system prefix: keeping the last two leaves exactly one to remove.
        messages = [
            ChatMessage.from_system("rules"),
            ChatMessage.from_user("a"),
            ChatMessage.from_user("b"),
            ChatMessage.from_user("c"),
        ]

        assert SlidingWindowCompactor(keep_last_n_messages=2).compact(_state(messages)) is None
        # Without a note there is a real saving, so the same cut is worth making.
        assert SlidingWindowCompactor(keep_last_n_messages=2, omission_note=False).compact(_state(messages)) == [
            messages[0],
            *messages[2:],
        ]

    def test_rejects_keeping_no_messages(self):
        with pytest.raises(ValueError, match="`keep_last_n_messages` must be at least 1"):
            SlidingWindowCompactor(keep_last_n_messages=0)

    def test_serde_round_trip(self):
        compactor = SlidingWindowCompactor(keep_last_n_messages=7, omission_note=False)
        data = compactor.to_dict()

        assert data == {
            "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
            "init_parameters": {"keep_last_n_messages": 7, "omission_note": False},
        }
        restored = SlidingWindowCompactor.from_dict(data)
        assert restored.keep_last_n_messages == 7
        assert restored.omission_note is False


class TestSlidingWindowCompactorAsync:
    @pytest.mark.asyncio
    async def test_compact_async_matches_compact(self):
        # `SlidingWindowCompactor` does no I/O, so it relies on the protocol's default `compact_async`.
        compactor = SlidingWindowCompactor(keep_last_n_messages=2)
        messages = _long_conversation()

        assert await compactor.compact_async(_state(messages)) == compactor.compact(_state(messages))

    @pytest.mark.asyncio
    async def test_compact_async_returns_none_when_nothing_to_remove(self):
        compactor = SlidingWindowCompactor(keep_last_n_messages=20)
        assert await compactor.compact_async(_state([ChatMessage.from_user("hi")])) is None
