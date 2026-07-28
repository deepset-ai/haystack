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
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.tools import tool

SCHEMA = {
    "messages": {"type": list[ChatMessage]},
    "step_count": {"type": int},
    "context_tokens": {"type": int},
    "token_usage": {"type": dict},
    "tool_call_counts": {"type": dict},
}

# `_record_context_tokens` sums the prompt and completion tokens, so every reply reports a context of 1000 tokens.
USAGE_META = {"usage": {"prompt_tokens": 900, "completion_tokens": 100}}


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
    base = {"messages": messages, "step_count": 2, "context_tokens": 900, "token_usage": {}, "tool_call_counts": {}}
    return State(schema=SCHEMA, data={**base, **data})


def _agent(hooks) -> Agent:
    return Agent(
        chat_generator=MockChatGenerator(
            responses=[_tool_call("c1"), _tool_call("c2"), _tool_call("c3"), "done"], meta=USAGE_META
        ),
        tools=[fetch],
        system_prompt="rules",
        hooks=hooks,
    )


def _assert_every_tool_result_is_answered(messages: list[ChatMessage]) -> None:
    """A tool result whose originating call is missing from the history is rejected by chat-completion APIs."""
    offered_call_ids: set[str | None] = set()
    for message in messages:
        for call in message.tool_calls:
            offered_call_ids.add(call.id)
        for result in message.tool_call_results:
            assert result.origin.id in offered_call_ids, f"orphaned tool result: {result.origin}"


class TestContextCompactionHook:
    def test_rejects_a_threshold_that_is_always_met(self):
        with pytest.raises(ValueError, match="`threshold_tokens` must be at least 1"):
            ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=0)

    @pytest.mark.parametrize(
        ("context_tokens", "should_compact"),
        [
            pytest.param(150, True, id="over"),
            pytest.param(100, True, id="at-threshold"),
            pytest.param(50, False, id="under"),
            # A Chat Generator that reports no usage leaves `context_tokens` at 0, so the threshold is never reached.
            pytest.param(0, False, id="usage-never-reported"),
        ],
    )
    def test_trigger(self, context_tokens, should_compact):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_tokens=100)

        hook.run(_state(_long_conversation(), context_tokens=context_tokens))

        assert compactor.calls == (["compact"] if should_compact else [])

    def test_rewrites_messages_and_resets_context_tokens(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=100)
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

    def test_leaves_the_conversation_alone_when_the_compactor_declines(self):
        messages = _long_conversation()
        hook = ContextCompactionHook(compactor=_RecordingCompactor(result=None), threshold_tokens=100)
        state = _state(messages, context_tokens=900)

        hook.run(state)

        assert state.data["messages"] == messages
        assert state.data["context_tokens"] == 900

    def test_warns_when_usage_is_never_reported(self, caplog):
        # `context_tokens` still 0 after the first call means the Chat Generator reports no usage, so this hook can
        # never fire. `step_count == 1` happens once per run, so the warning is not repeated every step.
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=100)

        with caplog.at_level(logging.WARNING):
            hook.run(_state(_long_conversation(), step_count=1, context_tokens=0))

        assert "does not report token usage" in caplog.text

    def test_warns_only_once_across_a_run(self, caplog):
        # The `step_count` values a `before_llm` hook actually observes over a four-step run, confirmed against the
        # Agent loop. Only one of them is 1, which is what keeps the warning from repeating every step.
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=100)

        with caplog.at_level(logging.WARNING):
            for step_count in (0, 1, 2, 3):
                hook.run(_state(_long_conversation(), step_count=step_count, context_tokens=0))

        assert caplog.text.count("does not report token usage") == 1

    @pytest.mark.parametrize(
        ("step_count", "context_tokens"),
        [
            pytest.param(0, 0, id="before-the-first-call"),
            pytest.param(1, 50, id="usage-reported-but-under-threshold"),
            pytest.param(2, 0, id="not-the-first-step"),
        ],
    )
    def test_does_not_warn_about_usage(self, caplog, step_count, context_tokens):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=100)

        with caplog.at_level(logging.WARNING):
            hook.run(_state(_long_conversation(), step_count=step_count, context_tokens=context_tokens))

        assert "does not report token usage" not in caplog.text

    def test_lifecycle_delegates_to_the_compactor(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_tokens=100)

        hook.warm_up()
        hook.close()

        assert compactor.calls == ["warm_up", "close"]

    def test_cannot_be_registered_at_another_hook_point(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=100)

        with pytest.raises(ValueError, match="before_llm"):
            Agent(chat_generator=MockChatGenerator(), tools=[fetch], hooks={"after_tool": [hook]})

    def test_serde_round_trip(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=4), threshold_tokens=100_000)
        data = hook.to_dict()

        assert data == {
            "type": "haystack.hooks.compaction.hooks.ContextCompactionHook",
            "init_parameters": {
                "compactor": {
                    "type": "haystack.hooks.compaction.sliding_window.SlidingWindowCompactor",
                    "init_parameters": {"keep_last_n_messages": 4, "omission_note": _DEFAULT_OMISSION_NOTE},
                },
                "threshold_tokens": 100_000,
            },
        }
        restored = ContextCompactionHook.from_dict(data)
        assert isinstance(restored.compactor, SlidingWindowCompactor)
        assert restored.compactor.keep_last_n_messages == 4
        assert restored.threshold_tokens == 100_000

    def test_survives_an_agent_serde_round_trip(self):
        agent = _agent(
            {
                "before_llm": [
                    ContextCompactionHook(
                        compactor=SlidingWindowCompactor(keep_last_n_messages=3), threshold_tokens=100
                    )
                ]
            }
        )

        restored = Agent.from_dict(agent.to_dict())

        hook = restored.hooks["before_llm"][0]
        assert isinstance(hook, ContextCompactionHook)
        assert isinstance(hook.compactor, SlidingWindowCompactor)
        assert hook.compactor.keep_last_n_messages == 3


class TestContextCompactionHookInAgent:
    def test_compacts_a_multi_step_run(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=1000)

        compacted = _agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])
        uncompacted = _agent(None).run(messages=[ChatMessage.from_user("start")])

        messages = compacted["messages"]
        assert len(messages) < len(uncompacted["messages"])
        # Exactly one omission note: each compaction folds the previous one into the block it drops.
        assert sum(_COMPACTION_META_KEY in message.meta for message in messages) == 1
        assert messages[0].text == "rules"
        assert compacted["last_message"].text == "done"
        _assert_every_tool_result_is_answered(messages)

    def test_does_not_compact_below_the_threshold(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=1_000_000)

        result = _agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])

        assert not any(_COMPACTION_META_KEY in message.meta for message in result["messages"])


class TestContextCompactionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_uses_the_async_compaction_path(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_tokens=100)

        await hook.run_async(_state(_long_conversation()))

        assert compactor.calls == ["compact_async"]

    @pytest.mark.asyncio
    async def test_run_async_rewrites_messages(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=100)
        state = _state(_long_conversation(), context_tokens=900)

        await hook.run_async(state)

        assert len(state.data["messages"]) == 4
        assert state.data["context_tokens"] == 0

    @pytest.mark.asyncio
    async def test_lifecycle_prefers_the_async_methods(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_tokens=100)

        await hook.warm_up_async()
        await hook.close_async()

        assert compactor.calls == ["warm_up_async", "close_async"]

    @pytest.mark.asyncio
    async def test_compacts_a_multi_step_async_run(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=1000)

        result = await _agent({"before_llm": [hook]}).run_async(messages=[ChatMessage.from_user("start")])

        assert sum(_COMPACTION_META_KEY in message.meta for message in result["messages"]) == 1
        _assert_every_tool_result_is_answered(result["messages"])
