# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.compaction import Compactor, ContextCompactionHook, SlidingWindowCompactor
from haystack.hooks.compaction.hooks import _conversation_chars
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.tools import tool

SCHEMA = {
    "messages": {"type": list[ChatMessage]},
    "step_count": {"type": int},
    "context_tokens": {"type": int},
    "token_usage": {"type": dict},
    "tool_call_counts": {"type": dict},
}


@tool
def fetch(topic: Annotated[str, "the topic to fetch"]) -> str:
    """Fetch a document about a topic."""
    return "DATA " * 200


class _RecordingCompactor(Compactor):
    """A compactor that returns a preset result and records every call made to it."""

    def __init__(self, result: list[ChatMessage] | None = None) -> None:
        self.result = result
        self.calls: list[str] = []

    def compact(self, state: State) -> list[ChatMessage] | None:
        self.calls.append("compact")
        return self.result

    async def compact_async(self, state: State) -> list[ChatMessage] | None:
        self.calls.append("compact_async")
        return self.result

    def warm_up(self) -> None:
        self.calls.append("warm_up")

    async def warm_up_async(self) -> None:
        self.calls.append("warm_up_async")

    def close(self) -> None:
        self.calls.append("close")

    async def close_async(self) -> None:
        self.calls.append("close_async")


def _tool_result(result: str, *, call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_tool(tool_result=result, origin=ToolCall(tool_name="fetch", arguments={}, id=call_id))


def _tool_call(call_id: str = "c1") -> ChatMessage:
    return ChatMessage.from_assistant(
        tool_calls=[ToolCall(tool_name="fetch", arguments={"topic": "haystack"}, id=call_id)]
    )


def _long_conversation() -> list[ChatMessage]:
    return [
        ChatMessage.from_system("rules"),
        ChatMessage.from_user("start"),
        _tool_call("c1"),
        _tool_result("R" * 400, call_id="c1"),
        _tool_call("c2"),
        _tool_result("R" * 400, call_id="c2"),
    ]


def _state(messages: list[ChatMessage], **data) -> State:
    base = {"messages": messages, "step_count": 2, "context_tokens": 0, "token_usage": {}, "tool_call_counts": {}}
    return State(schema=SCHEMA, data={**base, **data})


def _assert_every_tool_result_is_answered(messages: list[ChatMessage]) -> None:
    """A tool result whose originating call is missing from the history is rejected by chat-completion APIs."""
    offered_call_ids: set[str | None] = set()
    for message in messages:
        for call in message.tool_calls:
            offered_call_ids.add(call.id)
        for result in message.tool_call_results:
            assert result.origin.id in offered_call_ids, f"orphaned tool result: {result.origin}"


class TestConversationChars:
    @pytest.mark.parametrize(
        ("messages", "expected"),
        [
            pytest.param([], 0, id="empty"),
            pytest.param([ChatMessage.from_user("12345")], 5, id="message-text"),
            pytest.param(
                [_tool_call("c1")], len("fetch") + len('{"topic": "haystack"}'), id="tool-call-name-and-arguments"
            ),
            pytest.param([_tool_result("R" * 40)], 40, id="tool-result-content"),
        ],
    )
    def test_size(self, messages, expected):
        assert _conversation_chars(messages) == expected

    def test_shrinks_when_a_result_is_rewritten_in_place(self):
        # A strategy that replaces a result's content without removing the message must still register as a shrink,
        # which is why size is measured in characters rather than in messages.
        before = [_tool_call("c1"), _tool_result("R" * 500)]
        after = [_tool_call("c1"), _tool_result("[removed]")]
        assert len(after) == len(before)
        assert _conversation_chars(after) < _conversation_chars(before)


class TestContextCompactionHook:
    def test_requires_at_least_one_threshold(self):
        with pytest.raises(ValueError, match="at least one of `threshold_tokens` or `threshold_chars`"):
            ContextCompactionHook(compactor=SlidingWindowCompactor())

    @pytest.mark.parametrize(
        ("thresholds", "context_tokens", "should_compact"),
        [
            pytest.param({"threshold_tokens": 100}, 150, True, id="tokens-over"),
            pytest.param({"threshold_tokens": 100}, 100, True, id="tokens-at-threshold"),
            pytest.param({"threshold_tokens": 100}, 50, False, id="tokens-under"),
            # A Chat Generator that reports no usage leaves `context_tokens` at 0, so a token threshold never fires.
            pytest.param({"threshold_tokens": 100}, 0, False, id="tokens-never-reported"),
            pytest.param({"threshold_chars": 500}, 0, True, id="chars-over-without-usage"),
            pytest.param({"threshold_chars": 100_000}, 0, False, id="chars-under"),
            pytest.param({"threshold_tokens": 100, "threshold_chars": 100_000}, 150, True, id="either-fires"),
            pytest.param({"threshold_tokens": 100_000, "threshold_chars": 500}, 5, True, id="chars-fires-alone"),
        ],
    )
    def test_trigger(self, thresholds, context_tokens, should_compact):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, **thresholds)

        hook.run(_state(_long_conversation(), context_tokens=context_tokens))

        assert compactor.calls == (["compact"] if should_compact else [])

    def test_rewrites_messages_and_resets_context_tokens(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_chars=500)
        state = _state(
            _long_conversation(), context_tokens=900, token_usage={"prompt_tokens": 12}, tool_call_counts={"fetch": 2}
        )

        hook.run(state)

        messages = state.data["messages"]
        assert len(messages) == 4
        assert _COMPACTION_META_KEY in messages[1].meta
        # Reset to its "not yet measured" value; the next chat-generator call refreshes it from real usage.
        assert state.data["context_tokens"] == 0
        # Cumulative run metadata records what the run spent and did, which compaction does not change.
        assert state.data["token_usage"] == {"prompt_tokens": 12}
        assert state.data["tool_call_counts"] == {"fetch": 2}

    @pytest.mark.parametrize(
        "result",
        [pytest.param(None, id="compactor-declined"), pytest.param("unchanged", id="compactor-did-not-shrink")],
    )
    def test_leaves_the_conversation_alone_when_nothing_was_gained(self, result):
        messages = _long_conversation()
        compactor = _RecordingCompactor(result=list(messages) if result == "unchanged" else None)
        hook = ContextCompactionHook(compactor=compactor, threshold_chars=500)
        state = _state(messages, context_tokens=900)

        hook.run(state)

        assert state.data["messages"] == messages
        assert state.data["context_tokens"] == 900

    def test_warns_when_the_threshold_cannot_be_reached(self, caplog):
        # `threshold_chars` is below what the retained window costs, so compaction can never get under it.
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_chars=100)

        with caplog.at_level(logging.WARNING):
            hook.run(_state(_long_conversation()))

        assert "still" in caplog.text
        assert "`threshold_chars`" in caplog.text

    def test_lifecycle_delegates_to_the_compactor(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_chars=500)

        hook.warm_up()
        hook.close()

        assert compactor.calls == ["warm_up", "close"]

    def test_cannot_be_registered_at_another_hook_point(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_chars=500)

        with pytest.raises(ValueError, match="before_llm"):
            Agent(chat_generator=MockChatGenerator(), tools=[fetch], hooks={"after_tool": [hook]})

    def test_serde_round_trip(self):
        hook = ContextCompactionHook(
            compactor=SlidingWindowCompactor(keep_last_n_messages=4), threshold_tokens=100, threshold_chars=500
        )
        data = hook.to_dict()

        assert data == {
            "type": "haystack.hooks.compaction.hooks.ContextCompactionHook",
            "init_parameters": {
                "compactor": {
                    "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
                    "init_parameters": {"keep_last_n_messages": 4, "omission_note": True},
                },
                "threshold_tokens": 100,
                "threshold_chars": 500,
            },
        }
        restored = ContextCompactionHook.from_dict(data)
        assert isinstance(restored.compactor, SlidingWindowCompactor)
        assert restored.compactor.keep_last_n_messages == 4
        assert restored.threshold_tokens == 100

    def test_survives_an_agent_serde_round_trip(self):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[fetch],
            hooks={
                "before_llm": [
                    ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=3), threshold_chars=500)
                ]
            },
        )

        restored = Agent.from_dict(agent.to_dict())

        hook = restored.hooks["before_llm"][0]
        assert isinstance(hook, ContextCompactionHook)
        assert isinstance(hook.compactor, SlidingWindowCompactor)
        assert hook.compactor.keep_last_n_messages == 3


class TestContextCompactionHookInAgent:
    def _agent(self, hooks) -> Agent:
        return Agent(
            chat_generator=MockChatGenerator(responses=[_tool_call("c1"), _tool_call("c2"), _tool_call("c3"), "done"]),
            tools=[fetch],
            system_prompt="rules",
            hooks=hooks,
        )

    def test_compacts_a_multi_step_run(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_chars=1500)

        compacted = self._agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])
        uncompacted = self._agent(None).run(messages=[ChatMessage.from_user("start")])

        compacted_messages = compacted["messages"]
        assert len(compacted_messages) < len(uncompacted["messages"])
        # Exactly one omission note: each compaction folds the previous one into the block it drops.
        assert sum(_COMPACTION_META_KEY in message.meta for message in compacted_messages) == 1
        assert compacted_messages[0].text == "rules"
        assert compacted["last_message"].text == "done"
        _assert_every_tool_result_is_answered(compacted_messages)

    def test_does_not_compact_a_run_that_stays_small(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_chars=1_000_000)

        result = self._agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])

        assert not any(_COMPACTION_META_KEY in message.meta for message in result["messages"])


class TestContextCompactionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_uses_the_async_compaction_path(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_chars=500)

        await hook.run_async(_state(_long_conversation()))

        assert compactor.calls == ["compact_async"]

    @pytest.mark.asyncio
    async def test_run_async_rewrites_messages(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_chars=500)
        state = _state(_long_conversation(), context_tokens=900)

        await hook.run_async(state)

        assert len(state.data["messages"]) == 4
        assert state.data["context_tokens"] == 0

    @pytest.mark.asyncio
    async def test_lifecycle_prefers_the_async_methods(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_chars=500)

        await hook.warm_up_async()
        await hook.close_async()

        assert compactor.calls == ["warm_up_async", "close_async"]

    @pytest.mark.asyncio
    async def test_compacts_a_multi_step_async_run(self):
        agent = Agent(
            chat_generator=MockChatGenerator(responses=[_tool_call("c1"), _tool_call("c2"), _tool_call("c3"), "done"]),
            tools=[fetch],
            system_prompt="rules",
            hooks={
                "before_llm": [
                    ContextCompactionHook(
                        compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_chars=1500
                    )
                ]
            },
        )

        result = await agent.run_async(messages=[ChatMessage.from_user("start")])

        assert sum(_COMPACTION_META_KEY in message.meta for message in result["messages"]) == 1
        _assert_every_tool_result_is_answered(result["messages"])
