# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest

from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationUIResult,
    NeverAskPolicy,
    SimpleConsoleUI,
)
from haystack.human_in_the_loop.types import ConfirmationStrategy, ConfirmationUI
from haystack.tools import Tool, Toolset, create_tool_from_function


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
                            "async_function": None,
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
                "tool_concurrency_limit": 4,
                "tool_streaming_callback_passthrough": False,
                "required_variables": None,
                "user_prompt": None,
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
                "hooks": None,
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
        assert deserialized_agent.tool_concurrency_limit == agent.tool_concurrency_limit
        assert deserialized_agent.tool_streaming_callback_passthrough == agent.tool_streaming_callback_passthrough
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"], BlockingConfirmationStrategy)
        assert isinstance(
            deserialized_agent._confirmation_strategies["addition_tool"].confirmation_policy, NeverAskPolicy
        )
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"].confirmation_ui, SimpleConsoleUI)

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_blocking_confirmation_strategy_modify(self, tools):
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
        result = agent.run([ChatMessage.from_user("What is 2+2?")])

        assert isinstance(result["last_message"], ChatMessage)
        assert result["last_message"].text is not None
        assert "5" in result["last_message"].text
        # Auto-populated run-metadata outputs: at least one tool call plus a final answer.
        assert result["step_count"] >= 2
        assert result["tool_call_counts"]["addition_tool"] >= 1
        assert result["token_usage"]["prompt_tokens"] > 0
        assert result["token_usage"]["completion_tokens"] > 0
        assert result["token_usage"]["total_tokens"] > 0

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
        result = await agent.run_async([ChatMessage.from_user("What is 2+2?")])

        assert isinstance(result["last_message"], ChatMessage)
        assert result["last_message"].text is not None
        assert "5" in result["last_message"].text
        # Auto-populated run-metadata outputs: at least one tool call plus a final answer.
        assert result["step_count"] >= 2
        assert result["tool_call_counts"]["addition_tool"] >= 1
        assert result["token_usage"]["prompt_tokens"] > 0
        assert result["token_usage"]["completion_tokens"] > 0
        assert result["token_usage"]["total_tokens"] > 0


@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] | Toolset | None = None, **kwargs) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello")]}


def _producer() -> dict[str, str]:
    return {"value": "PRODUCED"}


def _consumer(value: Annotated[str, "the shared value"]) -> str:
    return f"consumed:{value}"


# `producer` overwrites the `shared` state key; `consumer` reads it via inputs_from_state. Called together in one
# step, the batched executor must run producer first so consumer sees the fresh value.
producer_tool = Tool(
    name="producer",
    description="Produce a value into state.",
    parameters={"type": "object", "properties": {}, "required": []},
    function=_producer,
    outputs_to_state={"shared": {"source": "value"}},
)
consumer_tool = Tool(
    name="consumer",
    description="Consume the shared value.",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    function=_consumer,
    inputs_from_state={"shared": "value"},
)


class TestConfirmationStrategyToolArgPrep:
    def test_confirmed_dependent_tool_runs_with_fresh_state(self):
        """
        When a confirmation strategy is configured, a tool that reads State a same-step tool writes must still run
        with the freshly-produced value, not the stale step-start value.
        """
        agent = Agent(
            chat_generator=MockChatGenerator(),
            tools=[producer_tool, consumer_tool],
            state_schema={"shared": {"type": str}},
            confirmation_strategies={
                "consumer": BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            },
        )
        agent.warm_up()
        # Step 1: the model calls producer and consumer together (consumer relies on inputs_from_state for `value`).
        # Step 2: a plain text reply ends the run.
        agent.chat_generator.run = MagicMock(
            side_effect=[
                {
                    "replies": [
                        ChatMessage.from_assistant(tool_calls=[ToolCall("producer", {}), ToolCall("consumer", {})])
                    ]
                },
                {"replies": [ChatMessage.from_assistant("done")]},
            ]
        )

        # `shared` starts stale; producer overwrites it to "PRODUCED" before consumer runs.
        result = agent.run(messages=[ChatMessage.from_user("go")], shared="OLD")

        consumer_results = [
            m.tool_call_result.result
            for m in result["messages"]
            if m.tool_call_result is not None and m.tool_call_result.origin.tool_name == "consumer"
        ]
        assert consumer_results == ["consumed:PRODUCED"]
