# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.hooks.budget import StepBudgetReminderHook, TokenBudgetHook
from haystack.hooks.budget.hooks import _FINAL_MESSAGE_TEXT
from haystack.hooks.utils import _SYSTEM_REMINDER_META_KEY
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
    schema = {
        "token_usage": {"type": dict[str, Any]},
        "stop_run": {"type": str},
        "messages": {"type": list[ChatMessage]},
    }
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
    def test_stops_when_usage_reaches_the_budget(self, usage, caplog):
        state = _state(usage)
        with caplog.at_level(logging.WARNING):
            TokenBudgetHook(max_total_tokens=100).run(state)
        assert state.data["stop_run"] == "token_budget_exceeded"
        assert state.data.get("messages") is None
        assert "token budget of 100 (100 used)" in caplog.text

    @pytest.mark.parametrize("usage", [{"total_tokens": 99}, {}], ids=["under-budget", "no-usage-reported"])
    def test_does_not_stop_below_the_budget(self, usage):
        state = _state(usage)
        TokenBudgetHook(max_total_tokens=100).run(state)
        assert state.data.get("stop_run") is None

    def test_adds_a_final_message(self):
        state = _state({"total_tokens": 100})
        TokenBudgetHook(max_total_tokens=100, add_final_message=True).run(state)
        assert state.data["messages"][-1].text == _FINAL_MESSAGE_TEXT
        assert state.data["messages"][-1].is_from(ChatRole.ASSISTANT)

    def test_non_positive_budget_raises(self):
        with pytest.raises(ValueError, match="max_total_tokens"):
            TokenBudgetHook(max_total_tokens=0)

    def test_to_dict_from_dict_roundtrip(self):
        hook = TokenBudgetHook(max_total_tokens=5000, add_final_message=True)
        restored = TokenBudgetHook.from_dict(hook.to_dict())
        assert restored.max_total_tokens == 5000
        assert restored.add_final_message is True

    def test_stops_an_agent_run_when_the_budget_is_spent(self):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[fetch],
            hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100)]},
        )
        agent.warm_up()
        agent.chat_generator.run = MagicMock(  # type: ignore[method-assign]
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

            def to_dict(self) -> dict[str, Any]:
                return default_to_dict(self)

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "KeepIterating":
                return default_from_dict(cls, data)

        agent = Agent(
            chat_generator=MockChatGenerator(),
            hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100)], "on_exit": [KeepIterating()]},
        )
        agent.warm_up()
        agent.chat_generator.run = MagicMock(  # type: ignore[method-assign]
            return_value={"replies": [ChatMessage.from_assistant("draft", meta={"usage": {"total_tokens": 60}})]}
        )
        result = agent.run(messages=[ChatMessage.from_user("hi")])
        assert agent.chat_generator.run.call_count == 2
        assert result["exit_reason"] == "token_budget_exceeded"


_TWO_LEFT = (
    "<system-reminder>\nYou have 2 of 4 steps remaining before this run is stopped. Stop gathering information and "
    "start working toward your final answer.\n</system-reminder>"
)
_LAST_STEP = (
    "<system-reminder>\nThis is your last step of 4. Give your final answer now: you will not see the results of any "
    "further tool calls.\n</system-reminder>"
)


def _step_state(step_count: int, max_agent_steps: int) -> State:
    schema = {"step_count": {"type": int}, "max_agent_steps": {"type": int}, "messages": {"type": list[ChatMessage]}}
    return State(schema=schema, data={"step_count": step_count, "max_agent_steps": max_agent_steps, "messages": []})


class TestStepBudgetReminderHook:
    @pytest.mark.parametrize(
        ("step_count", "expected"),
        [(1, None), (2, _TWO_LEFT), (3, _LAST_STEP)],
        ids=["outside-window", "2-left", "last"],
    )
    def test_reminds_within_the_window(self, step_count, expected):
        state = _step_state(step_count=step_count, max_agent_steps=4)
        StepBudgetReminderHook(remind_within_steps=2).run(state)
        if expected is None:
            assert state.data["messages"] == []
        else:
            [reminder] = state.data["messages"]
            assert reminder.is_from(ChatRole.USER)
            assert reminder.meta == {_SYSTEM_REMINDER_META_KEY: True}
            assert reminder.text == expected

    def test_non_positive_window_raises(self):
        with pytest.raises(ValueError, match="`remind_within_steps` must be a positive number of steps, got 0."):
            StepBudgetReminderHook(remind_within_steps=0)

    def test_to_dict_from_dict_roundtrip(self):
        restored = StepBudgetReminderHook.from_dict(StepBudgetReminderHook(remind_within_steps=5).to_dict())
        assert restored.remind_within_steps == 5

    def test_reminds_an_agent_before_its_last_llm_calls(self):
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[fetch],
            max_agent_steps=4,
            hooks={"before_llm": [StepBudgetReminderHook(remind_within_steps=2)]},
        )
        agent.warm_up()
        # Record the last message of each LLM call's input, since the Agent keeps appending to the conversation.
        last_messages: list[ChatMessage] = []

        def fake_run(messages: list[ChatMessage], **kwargs: Any) -> dict:
            last_messages.append(messages[-1])
            return {"replies": [ChatMessage.from_assistant(tool_calls=[ToolCall("fetch", {"topic": "x"})])]}

        agent.chat_generator.run = MagicMock(side_effect=fake_run)  # type: ignore[method-assign]
        result = agent.run(messages=[ChatMessage.from_user("hi")])

        assert result["exit_reason"] == "max_agent_steps"
        assert [message.text for message in last_messages] == ["hi", None, _TWO_LEFT, _LAST_STEP]
