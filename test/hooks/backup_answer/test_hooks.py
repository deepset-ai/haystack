# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

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
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="fetch", arguments={"topic": "x"})]),
        ChatMessage.from_tool("DATA", origin=ToolCall(tool_name="fetch", arguments={"topic": "x"})),
    ]


class TestBackupAnswerHook:
    def test_to_dict_and_from_dict(self):
        hook = BackupAnswerHook(
            chat_generator=MockChatGenerator(responses="backup"),
            system_prompt="custom",
            exit_reasons=("max_agent_steps", "token_budget_exceeded"),
            raise_on_failure=True,
        )
        restored = BackupAnswerHook.from_dict(hook.to_dict())
        assert isinstance(restored.chat_generator, MockChatGenerator)
        assert restored.system_prompt == "custom"
        assert restored.exit_reasons == ("max_agent_steps", "token_budget_exceeded")
        assert restored.raise_on_failure is True

    def test_appends_backup_answer_with_replaced_system_prompt(self):
        generator = MagicMock()
        generator.run.return_value = {"replies": [ChatMessage.from_assistant("backup")]}
        state = _state(_transcript(), exit_reason="max_agent_steps")

        BackupAnswerHook(chat_generator=generator).run(state)

        assert state.data["messages"][-1].text == "backup"
        assert state.data["exit_reason"] == "max_agent_steps"
        prompt = generator.run.call_args.kwargs["messages"]
        assert prompt[0].text == _DEFAULT_SYSTEM_PROMPT
        assert prompt[1].text == "question"
        assert not any(m.text == "agent prompt" for m in prompt)

    @pytest.mark.parametrize("exit_reason", ["text", "token_budget_exceeded", None])
    def test_skips_other_exit_reasons(self, exit_reason):
        state = _state(_transcript(), exit_reason=exit_reason)
        BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup")).run(state)
        assert len(state.data["messages"]) == 4

    def test_skips_conversation_without_non_system_messages(self):
        generator = MagicMock()
        state = _state([ChatMessage.from_system("agent prompt")], exit_reason="max_agent_steps")
        BackupAnswerHook(chat_generator=generator).run(state)
        generator.run.assert_not_called()

    def test_fires_on_configured_exit_reasons(self):
        state = _state(_transcript(), exit_reason="token_budget_exceeded")
        hook = BackupAnswerHook(
            chat_generator=MockChatGenerator(responses="backup"), exit_reasons=["token_budget_exceeded"]
        )
        hook.run(state)
        assert state.data["messages"][-1].text == "backup"

    def test_agent_writes_backup_answer_when_max_steps_reached(self):
        agent = Agent(
            chat_generator=MockChatGenerator(
                responses=[
                    ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="fetch", arguments={"topic": "x"})])
                ]
            ),
            tools=[fetch],
            max_agent_steps=2,
            hooks={"after_run": [BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup"))]},
        )
        result = agent.run(messages=[ChatMessage.from_user("question")])
        assert result["exit_reason"] == "max_agent_steps"
        assert result["last_message"].text == "backup"

    @pytest.mark.parametrize(
        "run_kwargs",
        [
            {"side_effect": RuntimeError("API down")},
            {"return_value": {"replies": []}},
            {"return_value": {"replies": [ChatMessage.from_assistant("")]}},
        ],
        ids=["generator_raises", "no_replies", "empty_reply"],
    )
    def test_failure_leaves_messages_unchanged(self, run_kwargs):
        generator = MagicMock()
        generator.run.configure_mock(**run_kwargs)
        state = _state(_transcript(), exit_reason="max_agent_steps")

        BackupAnswerHook(chat_generator=generator).run(state)

        assert state.data["messages"] == _transcript()

    def test_raise_on_failure(self):
        generator = MagicMock()
        generator.run.return_value = {"replies": []}
        hook = BackupAnswerHook(chat_generator=generator, raise_on_failure=True)
        with pytest.raises(RuntimeError, match="no usable text"):
            hook.run(_state(_transcript(), exit_reason="max_agent_steps"))

    def test_forwards_warm_up_and_close(self):
        generator = MagicMock()
        hook = BackupAnswerHook(chat_generator=generator)
        hook.warm_up()
        hook.close()
        generator.warm_up.assert_called_once()
        generator.close.assert_called_once()


class TestBackupAnswerHookAsync:
    @pytest.mark.asyncio
    async def test_run_async(self):
        state = _state(_transcript(), exit_reason="max_agent_steps")
        await BackupAnswerHook(chat_generator=MockChatGenerator(responses="backup")).run_async(state)
        assert state.data["messages"][-1].text == "backup"

    @pytest.mark.asyncio
    async def test_run_async_failure_leaves_messages_unchanged(self):
        generator = MagicMock(spec=["run", "run_async"])
        generator.run_async = AsyncMock(side_effect=RuntimeError("API down"))
        state = _state(_transcript(), exit_reason="max_agent_steps")

        await BackupAnswerHook(chat_generator=generator).run_async(state)

        assert state.data["messages"] == _transcript()

    @pytest.mark.asyncio
    async def test_forwards_async_warm_up_and_close(self):
        generator = MagicMock(spec=["run", "warm_up_async", "close_async"])
        generator.warm_up_async = AsyncMock()
        generator.close_async = AsyncMock()
        hook = BackupAnswerHook(chat_generator=generator)
        await hook.warm_up_async()
        await hook.close_async()
        generator.warm_up_async.assert_awaited_once()
        generator.close_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_warm_up_and_close_fall_back_to_sync(self):
        generator = MagicMock(spec=["run", "warm_up", "close"])
        hook = BackupAnswerHook(chat_generator=generator)
        await hook.warm_up_async()
        await hook.close_async()
        generator.warm_up.assert_called_once()
        generator.close.assert_called_once()
