# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, Toolset


@component
class LoopingChatGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "LoopingChatGenerator", "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopingChatGenerator":
        return cls()

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "replies": [
                ChatMessage.from_assistant(
                    "Calling the loop tool.",
                    tool_calls=[ToolCall(tool_name="loop", arguments={})],
                )
            ]
        }

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self.run(messages=messages, tools=tools, **kwargs)


def loop_tool() -> str:
    return "looped"


def make_agent() -> tuple[Agent, LoopingChatGenerator]:
    chat_generator = LoopingChatGenerator()
    tool = Tool(
        name="loop",
        description="Return a loop marker.",
        parameters={"type": "object", "properties": {}},
        function=loop_tool,
    )
    return (
        Agent(chat_generator=chat_generator, tools=[tool], max_agent_steps=10, max_agent_time_seconds=1.0),
        chat_generator,
    )


class TestAgentTimeLimit:
    def test_requires_positive_time_limit(self) -> None:
        with pytest.raises(ValueError, match="max_agent_time_seconds must be greater than zero"):
            Agent(chat_generator=LoopingChatGenerator(), max_agent_time_seconds=0)

    def test_serializes_time_limit(self) -> None:
        agent, _ = make_agent()

        assert agent.to_dict()["init_parameters"]["max_agent_time_seconds"] == 1.0

    def test_stops_before_tool_invocation_when_time_limit_expires(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent, chat_generator = make_agent()
        values = iter([10.0, 10.0, 11.0])
        monkeypatch.setattr("haystack.components.agents.agent.time.monotonic", lambda: next(values))

        with caplog.at_level(logging.WARNING):
            result = agent.run(messages=[ChatMessage.from_user("Start")])

        assert chat_generator.calls == 1
        assert result["last_message"].tool_calls[0].tool_name == "loop"
        assert "Agent reached maximum execution time of 1.0 seconds, stopping." in caplog.text

    @pytest.mark.asyncio
    async def test_stops_before_tool_invocation_when_time_limit_expires_async(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent, chat_generator = make_agent()
        values = iter([10.0, 10.0, 11.0])
        monkeypatch.setattr("haystack.components.agents.agent.time.monotonic", lambda: next(values))

        with caplog.at_level(logging.WARNING):
            result = await agent.run_async(messages=[ChatMessage.from_user("Start")])

        assert chat_generator.calls == 1
        assert result["last_message"].tool_calls[0].tool_name == "loop"
        assert "Agent reached maximum execution time of 1.0 seconds, stopping." in caplog.text
