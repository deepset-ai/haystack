# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction import Compactor, ContextCompactionHook, SlidingWindowCompactor
from haystack.hooks.compaction.sliding_window import _DEFAULT_OMISSION_NOTE
from haystack.tools import tool
from test.hooks.compaction.helpers import count_markers, long_conversation, make_state, tool_call


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


def _fetch_call(call_id: str) -> ChatMessage:
    """A call to the `fetch` tool, which the Agent-level tests actually have registered."""
    return tool_call(call_id, name="fetch", arguments={"topic": "haystack"})


def _agent(hooks) -> Agent:
    return Agent(
        chat_generator=MockChatGenerator(
            responses=[_fetch_call("c1"), _fetch_call("c2"), _fetch_call("c3"), "done"],
            meta={"usage": {"prompt_tokens": 900, "completion_tokens": 100}},
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
    def test_rejects_invalid_threshold(self):
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
        hook.run(make_state(long_conversation(), context_tokens=context_tokens))
        assert compactor.calls == (["compact"] if should_compact else [])

    def test_rewrites_messages_and_resets_context_tokens(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=100)
        state = make_state(
            long_conversation(), context_tokens=900, token_usage={"prompt_tokens": 12}, tool_call_counts={"fetch": 2}
        )
        hook.run(state)
        messages = state.data["messages"]
        assert len(messages) == 4
        assert count_markers(messages) == 1
        assert state.data["context_tokens"] == 0
        # Cumulative run metadata records what the run spent and did, which compaction does not change.
        assert state.data["token_usage"] == {"prompt_tokens": 12}
        assert state.data["tool_call_counts"] == {"fetch": 2}

    def test_leaves_the_conversation_alone_when_the_compactor_declines(self):
        messages = long_conversation()
        hook = ContextCompactionHook(compactor=_RecordingCompactor(result=None), threshold_tokens=100)
        state = make_state(messages, context_tokens=900)
        hook.run(state)
        assert state.data["messages"] == messages
        assert state.data["context_tokens"] == 900

    def test_warns_only_once_across_a_run(self, caplog):
        # The `step_count` values a `before_llm` hook actually observes over a four-step run, confirmed against the
        # Agent loop. Only one of them is 1, which is what keeps the warning from repeating every step.
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=100)
        with caplog.at_level(logging.WARNING):
            for step_count in (0, 1, 2, 3):
                hook.run(make_state(long_conversation(), step_count=step_count, context_tokens=0))
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
            hook.run(make_state(long_conversation(), step_count=step_count, context_tokens=context_tokens))
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


class TestContextCompactionHookInAgent:
    def test_compacts_a_multi_step_run(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=1000)
        compacted = _agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])
        uncompacted = _agent(None).run(messages=[ChatMessage.from_user("start")])
        messages = compacted["messages"]
        assert len(messages) < len(uncompacted["messages"])
        # Exactly one omission note: each compaction folds the previous one into the block it drops.
        assert count_markers(messages) == 1
        assert messages[0].text == "rules"
        assert compacted["last_message"].text == "done"
        _assert_every_tool_result_is_answered(messages)

    def test_does_not_compact_below_the_threshold(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(), threshold_tokens=1_000_000)
        result = _agent({"before_llm": [hook]}).run(messages=[ChatMessage.from_user("start")])
        assert count_markers(result["messages"]) == 0


class TestContextCompactionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_uses_the_async_compaction_path(self):
        compactor = _RecordingCompactor()
        hook = ContextCompactionHook(compactor=compactor, threshold_tokens=100)
        await hook.run_async(make_state(long_conversation()))
        assert compactor.calls == ["compact_async"]

    @pytest.mark.asyncio
    async def test_run_async_rewrites_messages(self):
        hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=2), threshold_tokens=100)
        state = make_state(long_conversation(), context_tokens=900)
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
        assert count_markers(result["messages"]) == 1
        _assert_every_tool_result_is_answered(result["messages"])
