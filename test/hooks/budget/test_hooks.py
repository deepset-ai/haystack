# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.budget import TokenBudgetHook
from haystack.tools import tool

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


@tool
def fetch(topic: Annotated[str, "the topic to fetch"]) -> str:
    """Fetch a document about a topic."""
    return "DATA"


def _fetch_reply(total_tokens: int) -> dict:
    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall("fetch", {"topic": "x"})], meta={"usage": {"total_tokens": total_tokens}}
    )
    return {"replies": [message]}


def _state(usage: dict) -> State:
    schema = {"token_usage": {"type": dict[str, Any]}, "stop_run": {"type": str}}
    return State(schema=schema, data={"token_usage": usage})


class TestTokenBudgetHook:
    @pytest.mark.parametrize(
        "usage",
        [
            {"total_tokens": 100},
            {"prompt_tokens": 60, "completion_tokens": 40},
            {"input_tokens": 60, "output_tokens": 40},
        ],
        ids=["total_tokens", "openai-style", "anthropic-style"],
    )
    def test_stops_when_usage_reaches_the_budget(self, usage):
        state = _state(usage)
        TokenBudgetHook(max_total_tokens=100).run(state)
        assert state.data["stop_run"] == "token_budget_exceeded"

    @pytest.mark.parametrize("usage", [{"total_tokens": 99}, {}], ids=["under-budget", "no-usage-reported"])
    def test_does_not_stop_below_the_budget(self, usage):
        state = _state(usage)
        TokenBudgetHook(max_total_tokens=100).run(state)
        assert state.data.get("stop_run") is None

    def test_non_positive_budget_raises(self):
        with pytest.raises(ValueError, match="max_total_tokens"):
            TokenBudgetHook(max_total_tokens=0)

    def test_to_dict_from_dict_roundtrip(self):
        restored = TokenBudgetHook.from_dict(TokenBudgetHook(max_total_tokens=5000).to_dict())
        assert restored.max_total_tokens == 5000

    def test_stops_an_agent_run_when_the_budget_is_spent(self):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[fetch],
            hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100)]},
        )
        agent.warm_up()
        agent.chat_generator.run = MagicMock(
            side_effect=[_fetch_reply(60), _fetch_reply(60), {"replies": [ChatMessage.from_assistant("done")]}]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 2
        assert result["tool_call_counts"]["fetch"] == 2
        assert result["exit_reason"] == "token_budget_exceeded"

    def test_stops_a_text_only_loop_kept_alive_by_continue_run(self):
        class KeepIterating:
            def run(self, state: State) -> None:
                state.set("continue_run", True)

        agent = Agent(
            chat_generator=MockChatGenerator(),
            hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100)], "on_exit": [KeepIterating()]},
        )
        agent.warm_up()
        agent.chat_generator.run = MagicMock(
            return_value={"replies": [ChatMessage.from_assistant("draft", meta={"usage": {"total_tokens": 60}})]}
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 2
        assert result["exit_reason"] == "token_budget_exceeded"
