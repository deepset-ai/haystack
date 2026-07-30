# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction import Compactor, ContextCompactionHook, SlidingWindowCompactor
from haystack.token_counters import TokenCounter
from haystack.tools import tool
from haystack.utils.experimental import ExperimentalWarning
from test.hooks.compaction.helpers import FakeCounter, count_markers, long_conversation, make_state, tool_call

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


def _hook(compactor=None, **overrides) -> ContextCompactionHook:
    settings = {"context_window": WINDOW, "compact_at": 0.7, "compact_to": 0.4, "token_counter": FakeCounter()}
    return ContextCompactionHook(compactor or SlidingWindowCompactor(min_keep_messages=2), **{**settings, **overrides})


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


class TestContextCompactionHookConfiguration:
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
        with pytest.warns(ExperimentalWarning, match="ContextCompactionHook.*experimental"):
            _hook()

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match="`context_window` must be a positive number of tokens"):
            _hook(context_window=0)

    def test_serde_round_trip(self):
        hook = ContextCompactionHook(
            compactor=SlidingWindowCompactor(min_keep_messages=4), context_window=200_000, compact_at=0.6
        )
        data = hook.to_dict()

        assert data["init_parameters"]["context_window"] == 200_000
        assert data["init_parameters"]["compact_at"] == 0.6
        assert data["init_parameters"]["compactor"]["init_parameters"]["min_keep_messages"] == 4
        assert data["init_parameters"]["token_counter"]["type"].endswith("ApproximateTokenCounter")

        restored = ContextCompactionHook.from_dict(data)
        assert isinstance(restored.compactor, SlidingWindowCompactor)
        assert restored.context_window == 200_000
        assert restored.compact_at == 0.6

    def test_survives_an_agent_serde_round_trip(self):
        agent = _agent({"before_llm": [_hook()]})

        hook = Agent.from_dict(agent.to_dict()).hooks["before_llm"][0]

        assert isinstance(hook, ContextCompactionHook)
        assert isinstance(hook.compactor, SlidingWindowCompactor)
        assert hook.context_window == WINDOW

    def test_cannot_be_registered_at_another_hook_point(self):
        with pytest.raises(ValueError, match="before_llm"):
            Agent(chat_generator=MockChatGenerator(), tools=[fetch], hooks={"after_tool": [_hook()]})


class TestContextCompactionHook:
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
        _hook(compactor).run(make_state(long_conversation(), context_tokens=context_tokens))

        assert compactor.calls == (["compact"] if should_compact else [])

    def test_fires_without_reported_usage_by_counting_locally(self):
        # A generator that reports no usage leaves `context_tokens` at 0. The conversation is then counted locally, so
        # compaction still works - which it did not before token counting.
        compactor = _RecordingCompactor()
        big = [ChatMessage.from_user("x" * 4000)] * 3
        _hook(compactor).run(make_state(big, context_tokens=0))

        assert compactor.calls == ["compact"]

    def test_hands_down_the_raw_target_when_there_is_no_overhead(self):
        # A conversation whose reported size matches what can be counted locally: nothing is held back. It has to end on
        # the assistant reply, which is what `context_tokens` is defined to account for.
        counter = FakeCounter()
        messages = [*(ChatMessage.from_user("x" * 400) for _ in range(9)), ChatMessage.from_assistant("y" * 400)]
        compactor = _RecordingCompactor()
        hook = _hook(compactor, token_counter=counter, context_window=1000)

        hook.run(make_state(messages, context_tokens=counter.count(messages)))

        assert compactor.targets[0] == pytest.approx(int(1000 * 0.4), abs=5)

    def test_subtracts_provider_overhead_from_the_target(self):
        # The reported count also covers tool schemas and template overhead, which a compactor cannot remove. Here that
        # overhead alone exceeds the target, so the compactor is told to cut the messages as far as it is allowed.
        compactor = _RecordingCompactor()
        _hook(compactor).run(make_state(long_conversation(), context_tokens=800))

        assert compactor.targets[0] == 0

    def test_rewrites_messages_and_re_estimates_context_tokens(self):
        hook = _hook()
        state = make_state(
            long_conversation(), context_tokens=800, token_usage={"prompt_tokens": 12}, tool_call_counts={"fetch": 2}
        )

        hook.run(state)

        assert len(state.data["messages"]) == 4
        assert count_markers(state.data["messages"]) == 1
        # Re-estimated rather than reset to 0, which would claim the context is empty.
        assert 0 < state.data["context_tokens"] < 800
        # Cumulative run metadata records what the run spent and did, which compaction does not change.
        assert state.data["token_usage"] == {"prompt_tokens": 12}
        assert state.data["tool_call_counts"] == {"fetch": 2}

    def test_leaves_the_conversation_alone_when_the_compactor_declines(self):
        messages = long_conversation()
        state = make_state(messages, context_tokens=800)

        _hook(_RecordingCompactor(result=None)).run(state)

        assert state.data["messages"] == messages
        assert state.data["context_tokens"] == 800

    def test_lifecycle_delegates_to_the_compactor(self):
        compactor = _RecordingCompactor()
        hook = _hook(compactor)

        hook.warm_up()
        hook.close()

        assert compactor.calls == ["warm_up", "close"]


class TestContextCompactionHookInAgent:
    def test_compacts_a_multi_step_run(self):
        compacted = _agent({"before_llm": [_hook()]}).run(messages=[ChatMessage.from_user("start")])
        uncompacted = _agent(None).run(messages=[ChatMessage.from_user("start")])

        messages = compacted["messages"]
        assert len(messages) < len(uncompacted["messages"])
        # Exactly one omission note: each compaction folds the previous one into the block it drops.
        assert count_markers(messages) == 1
        assert messages[0].text == "rules"
        assert compacted["last_message"].text == "done"
        _assert_every_tool_result_is_answered(messages)

    def test_does_not_compact_below_the_trigger(self):
        result = _agent({"before_llm": [_hook(context_window=10_000_000)]}).run(
            messages=[ChatMessage.from_user("start")]
        )

        assert count_markers(result["messages"]) == 0


class TestContextCompactionHookAsync:
    @pytest.mark.asyncio
    async def test_run_async_uses_the_async_compaction_path(self):
        compactor = _RecordingCompactor()
        await _hook(compactor).run_async(make_state(long_conversation(), context_tokens=800))

        assert compactor.calls == ["compact_async"]

    @pytest.mark.asyncio
    async def test_run_async_rewrites_messages(self):
        state = make_state(long_conversation(), context_tokens=800)

        await _hook().run_async(state)

        assert len(state.data["messages"]) == 4
        assert 0 < state.data["context_tokens"] < 800

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
