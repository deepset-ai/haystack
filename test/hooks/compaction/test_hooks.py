# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction import CompactionHook, Compactor, SlidingWindowCompactor, ToolResultPruningCompactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _estimated_context_tokens, _last_assistant_index
from haystack.hooks.invocation import _run_hooks, _run_hooks_async
from haystack.token_counters import TokenCounter
from haystack.tools import tool
from haystack.utils.experimental import ExperimentalWarning
from test.hooks.compaction.helpers import (
    FakeCounter,
    count_markers,
    fresh_conversation_with_two_steps,
    make_state,
    tool_call,
    tool_result,
)

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

WINDOW = 1000

# `_record_context_tokens` sums the prompt and completion tokens, so every reply reports a context of 800 tokens - 80%
# of the window, which is over any `compact_at` these tests use.
USAGE_META = {"usage": {"prompt_tokens": 700, "completion_tokens": 100}}


@tool
def fetch(topic: Annotated[str, "the topic to fetch"]) -> str:
    """Fetch a document about a topic."""
    return "DATA " * 200


class _RecordingCompactor(Compactor):
    """A compactor that returns a preset result and records every call made to it."""

    def __init__(self, result: list[ChatMessage] | None = None) -> None:
        self.result = result
        self.calls: list[str] = []
        self.targets: list[int] = []

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        self.calls.append("compact")
        self.targets.append(target_tokens)
        return self.result

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        self.calls.append("compact_async")
        self.targets.append(target_tokens)
        return self.result

    def warm_up(self) -> None:
        self.calls.append("warm_up")

    async def warm_up_async(self) -> None:
        self.calls.append("warm_up_async")

    def close(self) -> None:
        self.calls.append("close")

    async def close_async(self) -> None:
        self.calls.append("close_async")


def _hook(compactor=None, **overrides) -> CompactionHook:
    settings = {"context_window": WINDOW, "compact_at": 0.7, "compact_to": 0.4, "token_counter": FakeCounter()}
    return CompactionHook(compactor or SlidingWindowCompactor(), **{**settings, **overrides})


def _fetch_call(call_id: str) -> ChatMessage:
    """A call to the `fetch` tool, which the Agent-level tests actually have registered."""
    return tool_call(call_id, name="fetch", arguments={"topic": "haystack"})


def _agent(hooks) -> Agent:
    return Agent(
        chat_generator=MockChatGenerator(
            responses=[_fetch_call("c1"), _fetch_call("c2"), _fetch_call("c3"), "done"], meta=USAGE_META
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


class TestCompactionHookConfiguration:
    @pytest.mark.parametrize(
        ("compact_at", "compact_to"),
        [
            pytest.param(0.7, 0.7, id="target-equals-trigger"),
            pytest.param(0.4, 0.7, id="target-above-trigger"),
            pytest.param(1.5, 0.4, id="trigger-above-the-window"),
            pytest.param(0.7, 0.0, id="target-of-zero"),
        ],
    )
    def test_rejects_fractions_that_cannot_work(self, compact_at, compact_to):
        with pytest.raises(ValueError, match="0 < compact_to < compact_at <= 1"):
            _hook(compact_at=compact_at, compact_to=compact_to)

    @pytest.mark.filterwarnings("always::haystack.utils.experimental.ExperimentalWarning")
    def test_warns_that_the_feature_is_experimental(self):
        with pytest.warns(ExperimentalWarning, match="CompactionHook.*experimental"):
            _hook()

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match="`context_window` must be a positive number of tokens"):
            _hook(context_window=0)

    def test_serde_round_trip(self):
        hook = CompactionHook(
            compactor=SlidingWindowCompactor(min_keep_steps=4), context_window=200_000, compact_at=0.6
        )
        data = hook.to_dict()
        assert data["init_parameters"]["context_window"] == 200_000
        assert data["init_parameters"]["compact_at"] == 0.6
        assert data["init_parameters"]["compactor"]["init_parameters"]["min_keep_steps"] == 4
        assert data["init_parameters"]["token_counter"]["type"].endswith("ApproximateTokenCounter")
        restored = CompactionHook.from_dict(data)
        assert isinstance(restored.compactor, SlidingWindowCompactor)
        assert restored.context_window == 200_000
        assert restored.compact_at == 0.6

    def test_survives_an_agent_serde_round_trip(self):
        agent = _agent({"before_llm": [_hook()]})
        hook = Agent.from_dict(agent.to_dict()).hooks["before_llm"][0]
        assert isinstance(hook, CompactionHook)
        assert isinstance(hook.compactor, SlidingWindowCompactor)
        assert hook.context_window == WINDOW

    def test_cannot_be_registered_at_another_hook_point(self):
        with pytest.raises(ValueError, match="before_llm"):
            Agent(chat_generator=MockChatGenerator(), tools=[fetch], hooks={"after_tool": [_hook()]})


class TestCompactionHook:
    @pytest.mark.parametrize(
        ("context_tokens", "should_compact"),
        [
            pytest.param(800, True, id="over"),
            pytest.param(700, True, id="at-the-trigger"),
            pytest.param(300, False, id="under"),
        ],
    )
    def test_trigger(self, context_tokens, should_compact):
        compactor = _RecordingCompactor()
        _hook(compactor).run(make_state(messages=fresh_conversation_with_two_steps(), context_tokens=context_tokens))
        assert compactor.calls == (["compact"] if should_compact else [])

    def test_fires_without_reported_usage_by_counting_locally(self):
        # A generator that reports no usage leaves `context_tokens` at 0. The conversation is then counted locally, so
        # compaction still works - which it did not before token counting.
        compactor = _RecordingCompactor()
        big = [ChatMessage.from_user("x" * 4000)] * 3
        _hook(compactor).run(make_state(messages=big, context_tokens=0))
        assert compactor.calls == ["compact"]

    def test_counts_tool_schemas_without_reported_usage(self):
        counter = FakeCounter()
        messages = [ChatMessage.from_user("start")]
        total_tokens = counter.count(messages=messages, tools=[fetch])
        assert counter.count(messages) < total_tokens
        compactor = _RecordingCompactor()
        _hook(compactor, token_counter=counter, context_window=total_tokens, compact_at=1.0, compact_to=0.4).run(
            make_state(messages=messages, context_tokens=0, tools=[fetch])
        )
        assert compactor.calls == ["compact"]

    def test_hands_down_the_raw_target_when_there_is_no_overhead(self):
        # A conversation whose reported size matches what can be counted locally: nothing is held back. It has to end on
        # the assistant reply, which is what `context_tokens` is defined to account for.
        counter = FakeCounter()
        messages = [*(ChatMessage.from_user("x" * 400) for _ in range(9)), ChatMessage.from_assistant("y" * 400)]
        compactor = _RecordingCompactor()
        hook = _hook(compactor, token_counter=counter, context_window=1000)
        hook.run(make_state(messages=messages, context_tokens=counter.count(messages)))
        assert compactor.targets[0] == pytest.approx(int(1000 * 0.4), abs=5)

    def test_subtracts_provider_overhead_from_the_target(self):
        # The reported count also covers tool schemas and template overhead, which a compactor cannot remove. Here that
        # overhead alone exceeds the target, so the compactor is told to cut the messages as far as it is allowed.
        compactor = _RecordingCompactor()
        _hook(compactor).run(make_state(messages=fresh_conversation_with_two_steps(), context_tokens=800))
        assert compactor.targets[0] == 0

    def test_warns_when_the_token_counter_exceeds_the_context_estimate(self, caplog):
        messages = [ChatMessage.from_assistant("x" * 4000)]
        with caplog.at_level(logging.WARNING):
            _hook(_RecordingCompactor()).run(make_state(messages=messages, context_tokens=800))
        assert "TokenCounter estimated more tokens for the messages" in caplog.text

    def test_rewrites_messages_and_re_estimates_context_tokens(self):
        counter = FakeCounter()
        hook = _hook(token_counter=counter)
        messages = fresh_conversation_with_two_steps()
        original_context_tokens = 800
        estimated = _estimated_context_tokens(
            messages=messages, context_tokens=original_context_tokens, token_counter=counter
        )
        estimated_overhead = estimated - counter.count(messages=messages)
        state = make_state(
            messages=messages,
            context_tokens=original_context_tokens,
            token_usage={"prompt_tokens": 12},
            tool_call_counts={"fetch": 2},
        )
        hook.run(state=state)
        assert len(state.data["messages"]) == 5
        assert count_markers(state.data["messages"]) == 1
        last_assistant = _last_assistant_index(messages=state.data["messages"])
        expected = counter.count(messages=state.data["messages"][: last_assistant + 1]) + estimated_overhead
        assert state.data["context_tokens"] == expected
        assert 0 < state.data["context_tokens"] < original_context_tokens
        # Cumulative run metadata records what the run spent and did, which compaction does not change.
        assert state.data["token_usage"] == {"prompt_tokens": 12}
        assert state.data["tool_call_counts"] == {"fetch": 2}

    def test_preserves_the_no_usage_sentinel_after_compaction(self):
        compacted = [ChatMessage.from_assistant("kept")]
        state = make_state([ChatMessage.from_user("x" * 4000)], context_tokens=0)
        _hook(_RecordingCompactor(result=compacted)).run(state=state)
        assert state.data["messages"] == compacted
        assert state.data["context_tokens"] == 0

    def test_leaves_the_conversation_alone_when_the_compactor_declines(self):
        messages = fresh_conversation_with_two_steps()
        state = make_state(messages, context_tokens=800)
        _hook(_RecordingCompactor(result=None)).run(state=state)
        assert state.data["messages"] == messages
        assert state.data["context_tokens"] == 800

    def test_chains_tool_result_pruning_before_sliding_window(self):
        counter = FakeCounter(chars_per_token=1)
        messages = [
            ChatMessage.from_user("task"),
            tool_call("old"),
            tool_result("old result " * 400, call_id="old"),
            tool_call("recent"),
            tool_result("recent result " * 400, call_id="recent"),
        ]
        settings = {"context_window": 2000, "compact_at": 0.5, "compact_to": 0.1, "token_counter": counter}
        pruning_hook = CompactionHook(compactor=ToolResultPruningCompactor(min_keep_steps=1, min_tokens=0), **settings)
        sliding_window_hook = CompactionHook(compactor=SlidingWindowCompactor(), **settings)
        # Provider usage covers through the last assistant call plus request overhead; the trailing result is local.
        reported_context_tokens = counter.count(messages=messages[:-1]) + 100
        state = make_state(messages, context_tokens=reported_context_tokens)

        pruning_hook.run(state=state)
        after_pruning = state.data["messages"]
        assert after_pruning[2].meta[_COMPACTION_META_KEY]["strategy"] == "tool_result_pruning"
        assert len(after_pruning) == len(messages)

        # Pruning cannot reach the target while retaining the recent result, so the next hook sees the smaller history
        # but remains above its trigger and falls back to dropping the old step.
        sliding_window_hook.run(state=state)
        compacted = state.data["messages"]
        assert len(compacted) < len(after_pruning)
        assert any(
            message.meta.get(_COMPACTION_META_KEY, {}).get("strategy") == "sliding_window" for message in compacted
        )
        assert compacted[-2:] == messages[-2:]

    def test_lifecycle_delegates_to_the_compactor(self):
        compactor = _RecordingCompactor()
        hook = _hook(compactor)
        hook.warm_up()
        hook.close()
        assert compactor.calls == ["warm_up", "close"]


class TestCompactionHookInAgent:
    def test_compacts_a_multi_step_run(self):
        compacted = _agent({"before_llm": [_hook()]}).run(messages=[ChatMessage.from_user("start")])
        uncompacted = _agent(None).run(messages=[ChatMessage.from_user("start")])
        messages = compacted["messages"]
        assert len(messages) < len(uncompacted["messages"])
        # Exactly one omission note: each compaction folds the previous one into the block it drops.
        assert count_markers(messages=messages) == 1
        assert messages[0].text == "rules"
        assert compacted["last_message"].text == "done"
        _assert_every_tool_result_is_answered(messages)

    def test_does_not_compact_below_the_trigger(self):
        result = _agent({"before_llm": [_hook(context_window=10_000_000)]}).run(
            messages=[ChatMessage.from_user("start")]
        )
        assert count_markers(messages=result["messages"]) == 0


class TestCompactionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_uses_the_async_compaction_path(self):
        compactor = _RecordingCompactor()
        await _hook(compactor).run_async(make_state(fresh_conversation_with_two_steps(), context_tokens=800))
        assert compactor.calls == ["compact_async"]

    @pytest.mark.asyncio
    async def test_lifecycle_prefers_the_async_methods(self):
        compactor = _RecordingCompactor()
        hook = _hook(compactor)
        await hook.warm_up_async()
        await hook.close_async()
        assert compactor.calls == ["warm_up_async", "close_async"]

    @pytest.mark.asyncio
    async def test_compacts_a_multi_step_async_run(self):
        result = await _agent({"before_llm": [_hook()]}).run_async(messages=[ChatMessage.from_user("start")])
        assert count_markers(result["messages"]) == 1
        _assert_every_tool_result_is_answered(result["messages"])


class TestCompactionHookTracing:
    def test_adds_compaction_tags_to_hook_span(self, spying_tracer):
        compacted = [ChatMessage.from_assistant(text="kept")]
        hook = _hook(compactor=_RecordingCompactor(result=compacted))
        state = make_state(messages=[ChatMessage.from_assistant(text="original")], context_tokens=800)
        _run_hooks(hooks={"before_llm": [hook]}, hook_point="before_llm", state=state)
        span = spying_tracer.spans[0]
        assert span.operation_name == "haystack.agent.hook"
        assert span.parent_span is None
        assert span.tags == {
            # From _run_hooks
            "haystack.agent.hook.point": "before_llm",
            "haystack.agent.hook.name": "CompactionHook",
            "haystack.agent.hook.type": "haystack.hooks.compaction.hooks.CompactionHook",
            # From CompactionHook
            "haystack.agent.hook.compaction.strategy": "test.hooks.compaction.test_hooks._RecordingCompactor",
            "haystack.agent.hook.compaction.estimated_context_tokens": 800,
            "haystack.agent.hook.compaction.triggered": True,
            "haystack.agent.hook.compaction.target_tokens": 0,
            "haystack.agent.hook.compaction.compacted": True,
        }

    def test_traces_when_compaction_is_not_triggered(self, spying_tracer):
        hook = _hook(compactor=_RecordingCompactor())
        state = make_state(messages=[ChatMessage.from_assistant(text="original")], context_tokens=300)
        _run_hooks(hooks={"before_llm": [hook]}, hook_point="before_llm", state=state)
        span = spying_tracer.spans[0]
        assert span.operation_name == "haystack.agent.hook"
        assert span.parent_span is None
        assert span.tags == {
            # From _run_hooks
            "haystack.agent.hook.point": "before_llm",
            "haystack.agent.hook.name": "CompactionHook",
            "haystack.agent.hook.type": "haystack.hooks.compaction.hooks.CompactionHook",
            # From CompactionHook
            "haystack.agent.hook.compaction.strategy": "test.hooks.compaction.test_hooks._RecordingCompactor",
            "haystack.agent.hook.compaction.estimated_context_tokens": 300,
            "haystack.agent.hook.compaction.triggered": False,
        }

    @pytest.mark.asyncio
    async def test_adds_compaction_tags_to_hook_span_async(self, spying_tracer):
        compacted = [ChatMessage.from_assistant(text="kept")]
        hook = _hook(compactor=_RecordingCompactor(result=compacted))
        state = make_state(messages=[ChatMessage.from_assistant(text="original")], context_tokens=800)
        await _run_hooks_async(hooks={"before_llm": [hook]}, hook_point="before_llm", state=state)
        span = spying_tracer.spans[0]
        assert span.operation_name == "haystack.agent.hook"
        assert span.parent_span is None
        assert span.tags == {
            # From _run_hooks_async
            "haystack.agent.hook.point": "before_llm",
            "haystack.agent.hook.name": "CompactionHook",
            "haystack.agent.hook.type": "haystack.hooks.compaction.hooks.CompactionHook",
            # From CompactionHook
            "haystack.agent.hook.compaction.strategy": "test.hooks.compaction.test_hooks._RecordingCompactor",
            "haystack.agent.hook.compaction.estimated_context_tokens": 800,
            "haystack.agent.hook.compaction.triggered": True,
            "haystack.agent.hook.compaction.target_tokens": 0,
            "haystack.agent.hook.compaction.compacted": True,
        }
