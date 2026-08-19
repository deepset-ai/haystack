# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
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


def _agent(max_total_tokens: int) -> Agent:
    agent = Agent(
        chat_generator=MockChatGenerator(),
        tools=[fetch],
        hooks={"after_tool": [TokenBudgetHook(max_total_tokens=max_total_tokens)]},
    )
    agent.warm_up()
    return agent


class TestTokenBudgetHook:
    def test_stops_the_run_when_the_budget_is_reached(self):
        agent = _agent(max_total_tokens=100)
        agent.chat_generator.run = MagicMock(
            side_effect=[_fetch_reply(60), _fetch_reply(60), {"replies": [ChatMessage.from_assistant("done")]}]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 2
        assert result["exit_reason"] == "token_budget_exceeded"
        assert result["tool_call_counts"]["fetch"] == 2
        # The stopped run keeps the whole conversation: every requested tool call has its result.
        requested = [tc for m in result["messages"] for tc in m.tool_calls]
        answered = [m.tool_call_result.origin for m in result["messages"] if m.tool_call_result is not None]
        assert answered == requested

    def test_run_ends_normally_under_budget(self):
        agent = _agent(max_total_tokens=100)
        agent.chat_generator.run = MagicMock(
            side_effect=[
                _fetch_reply(10),
                {"replies": [ChatMessage.from_assistant("done", meta={"usage": {"total_tokens": 10}})]},
            ]
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert result["exit_reason"] == "text"

    def test_rejected_outside_after_tool(self):
        with pytest.raises(ValueError, match="after_tool"):
            Agent(chat_generator=MockChatGenerator(), hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100)]})

    def test_non_positive_budget_raises(self):
        with pytest.raises(ValueError, match="max_total_tokens"):
            TokenBudgetHook(max_total_tokens=0)

    def test_to_dict_from_dict_roundtrip(self):
        restored = TokenBudgetHook.from_dict(TokenBudgetHook(max_total_tokens=5000).to_dict())
        assert restored.max_total_tokens == 5000
