# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import MagicMock

import pytest

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.backup_answer import BackupAnswerHook
from haystack.hooks.backup_answer.hooks import _DEFAULT_SYSTEM_PROMPT
from haystack.tools import tool

pytestmark = pytest.mark.filterwarnings("ignore::haystack.utils.experimental.ExperimentalWarning")


@tool
def fetch(topic: Annotated[str, "the topic to fetch"]) -> str:
    """Fetch a document about a topic."""
    return "DATA"


def _state(messages: list[ChatMessage], exit_reason: str | None) -> State:
    schema = {"messages": {"type": list[ChatMessage]}, "exit_reason": {"type": str}}
    return State(schema=schema, data={"messages": messages, "exit_reason": exit_reason})


def _transcript() -> list[ChatMessage]:
    return [
        ChatMessage.from_system("agent prompt"),
        ChatMessage.from_user("question"),
        ChatMessage.from_assistant(tool_calls=[ToolCall("fetch", {"topic": "x"})]),
        ChatMessage.from_tool("DATA", origin=ToolCall("fetch", {"topic": "x"})),
    ]


class TestBackupAnswerHook:
    def test_to_dict_and_from_dict(self):
        hook = BackupAnswerHook(
            chat_generator=MockChatGenerator(responses="backup"),
            system_prompt="custom",
            exit_reasons=("max_agent_steps", "token_budget_exceeded"),
        )
        restored = BackupAnswerHook.from_dict(hook.to_dict())
        assert isinstance(restored.chat_generator, MockChatGenerator)
        assert restored.system_prompt == "custom"
        assert restored.exit_reasons == ("max_agent_steps", "token_budget_exceeded")

    def test_appends_backup_answer_with_replaced_system_prompt(self):
        generator = MagicMock()
        generator.run.return_value = {"replies": [ChatMessage.from_assistant("backup")]}
        state = _state(_transcript(), exit_reason="max_agent_steps")

        BackupAnswerHook(chat_generator=generator).run(state)

        assert state.data["messages"][-1].text == "backup"
        assert state.data["exit_reason"] == "max_agent_steps"
        prompt = generator.run.call_args.kwargs["messages"]
        assert prompt[0].text == _DEFAULT_SYSTEM_PROMPT
        assert [m.text for m in prompt[1:2]] == ["question"]
        assert not any(m.text == "agent prompt" for m in prompt)

    @pytest.mark.parametrize("exit_reason", ["text", "token_budget_exceeded", None])
    def test_skips_other_exit_reasons(self, exit_reason):
        state = _state(_transcript(), exit_reason=exit_reason)
        BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup")).run(state)
        assert len(state.data["messages"]) == 4

    def test_fires_on_configured_exit_reasons(self):
        state = _state(_transcript(), exit_reason="token_budget_exceeded")
        hook = BackupAnswerHook(
            chat_generator=MockChatGenerator(responses="backup"), exit_reasons=["token_budget_exceeded"]
        )
        hook.run(state)
        assert state.data["messages"][-1].text == "backup"

    @pytest.mark.asyncio
    async def test_run_async(self):
        state = _state(_transcript(), exit_reason="max_agent_steps")
        await BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup")).run_async(state)
        assert state.data["messages"][-1].text == "backup"

    def test_agent_writes_backup_answer_when_max_steps_reached(self):
        agent = Agent(
            chat_generator=MockChatGenerator(
                responses=[ChatMessage.from_assistant(tool_calls=[ToolCall("fetch", {"topic": "x"})])]
            ),
            tools=[fetch],
            max_agent_steps=2,
            hooks={"after_run": [BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup"))]},
        )
        result = agent.run(messages=[ChatMessage.from_user("question")])
        assert result["exit_reason"] == "max_agent_steps"
        assert result["last_message"].text == "backup"

    @pytest.mark.asyncio
    async def test_agent_run_async_writes_backup_answer_when_max_steps_reached(self):
        agent = Agent(
            chat_generator=MockChatGenerator(
                responses=[ChatMessage.from_assistant(tool_calls=[ToolCall("fetch", {"topic": "x"})])]
            ),
            tools=[fetch],
            max_agent_steps=2,
            hooks={"after_run": [BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup"))]},
        )
        result = await agent.run_async(messages=[ChatMessage.from_user("question")])
        assert result["last_message"].text == "backup"
