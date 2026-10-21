# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.hooks.system_reminders import StepBudgetReminderHook
from haystack.hooks.utils import _SYSTEM_REMINDER_META_KEY
from haystack.tools import tool

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")

_TWO_LEFT = (
    "<system-reminder>\nYou have 2 of 4 steps remaining before this run is stopped. Stop gathering information and "
    "start working toward your final answer.\n</system-reminder>"
)
_LAST_STEP = (
    "<system-reminder>\nThis is your last step of 4. Give your final answer now: you will not see the results of any "
    "further tool calls.\n</system-reminder>"
)


def _state(step_count: int, max_agent_steps: int) -> State:
    schema = {"step_count": {"type": int}, "max_agent_steps": {"type": int}, "messages": {"type": list[ChatMessage]}}
    return State(schema=schema, data={"step_count": step_count, "max_agent_steps": max_agent_steps, "messages": []})


@tool
def fetch(topic: Annotated[str, "the topic to fetch"]) -> str:
    """Fetch a document about a topic."""
    return "DATA"


class TestStepBudgetReminderHook:
    @pytest.mark.parametrize(
        ("step_count", "expected"),
        [(1, None), (2, _TWO_LEFT), (3, _LAST_STEP)],
        ids=["outside-window", "2-left", "last"],
    )
    def test_reminds_within_the_window(self, step_count, expected):
        state = _state(step_count=step_count, max_agent_steps=4)
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
