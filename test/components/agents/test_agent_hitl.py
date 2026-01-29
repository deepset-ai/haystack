# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationUIResult,
    NeverAskPolicy,
    SimpleConsoleUI,
)
from haystack.human_in_the_loop.types import ConfirmationStrategy, ConfirmationUI
from haystack.tools import Tool, create_tool_from_function


class MockUserInterface(ConfirmationUI):
    def __init__(self, ui_result: ConfirmationUIResult) -> None:
        self.ui_result = ui_result

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        return self.ui_result


def addition_tool(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def tools() -> list[Tool]:
    tool = create_tool_from_function(
        function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
    )
    return [tool]


@pytest.fixture
def confirmation_strategies() -> dict[str, ConfirmationStrategy]:
    return {
        "addition_tool": BlockingConfirmationStrategy(
            confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )
    }


class TestAgent:
    def test_to_dict(self, tools, confirmation_strategies, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(), tools=tools, confirmation_strategies=confirmation_strategies
        )
        agent_dict = agent.to_dict()
        assert agent_dict == {
            "type": "haystack.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-5-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "addition_tool",
                            "description": "A tool that adds two integers together.",
                            "parameters": {
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                                "type": "object",
                            },
                            "function": "test_agent_hitl.addition_tool",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    }
                ],
                "system_prompt": None,
                "exit_conditions": ["text"],
                "state_schema": {},
                "max_agent_steps": 100,
                "streaming_callback": None,
                "raise_on_tool_invocation_failure": False,
                "tool_invoker_kwargs": None,
                "confirmation_strategies": {
                    "addition_tool": {
                        "type": "haystack.human_in_the_loop.strategies.BlockingConfirmationStrategy",
                        "init_parameters": {
                            "confirmation_policy": {
                                "type": "haystack.human_in_the_loop.policies.NeverAskPolicy",
                                "init_parameters": {},
                            },
                            "confirmation_ui": {
                                "type": "haystack.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                                "init_parameters": {},
                            },
                            "reject_template": "Tool execution for '{tool_name}' was rejected by the user.",
                            "modify_template": "The parameters for tool '{tool_name}' were updated by the user to:"
                            "\n{final_tool_params}",
                            "user_feedback_template": "With user feedback: {feedback}",
                        },
                    }
                },
            },
        }

    def test_from_dict(self, tools, confirmation_strategies, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(), tools=tools, confirmation_strategies=confirmation_strategies
        )
        deserialized_agent = Agent.from_dict(agent.to_dict())
        assert deserialized_agent.to_dict() == agent.to_dict()
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert len(deserialized_agent.tools) == 1
        assert deserialized_agent.tools[0].name == "addition_tool"
        assert isinstance(deserialized_agent._tool_invoker, type(agent._tool_invoker))
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"], BlockingConfirmationStrategy)
        assert isinstance(
            deserialized_agent._confirmation_strategies["addition_tool"].confirmation_policy, NeverAskPolicy
        )
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"].confirmation_ui, SimpleConsoleUI)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_blocking_confirmation_strategy_modify(self, tools):
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=AlwaysAskPolicy(),
                    confirmation_ui=MockUserInterface(
                        ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3})
                    ),
                )
            },
        )
        agent.warm_up()

        result = agent.run([ChatMessage.from_user("What is 2+2?")])

        assert isinstance(result["last_message"], ChatMessage)
        assert result["last_message"].text is not None
        assert "5" in result["last_message"].text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_blocking_confirmation_strategy_modify(self, tools):
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=AlwaysAskPolicy(),
                    confirmation_ui=MockUserInterface(
                        ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3})
                    ),
                )
            },
        )
        agent.warm_up()

        result = await agent.run_async([ChatMessage.from_user("What is 2+2?")])

        assert isinstance(result["last_message"], ChatMessage)
        assert result["last_message"].text is not None
        assert "5" in result["last_message"].text
