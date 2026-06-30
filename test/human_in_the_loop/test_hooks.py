# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import merge_lists, replace_values
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    ConfirmationUIResult,
    NeverAskPolicy,
    SimpleConsoleUI,
)
from haystack.human_in_the_loop.types import ConfirmationUI
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
    return [
        create_tool_from_function(
            function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
        )
    ]


def _state_with(messages: list[ChatMessage], tools: list[Tool]) -> State:
    schema = {
        "messages": {"type": list[ChatMessage], "handler": merge_lists},
        "tools": {"type": list, "handler": replace_values},
    }
    state = State(schema=schema, data={})
    state.set("messages", messages, handler_override=replace_values)
    state.set("tools", tools, handler_override=replace_values)
    return state


def _confirm_hook(ui_result: ConfirmationUIResult) -> ConfirmationHook:
    return ConfirmationHook(
        confirmation_strategies={
            "addition_tool": BlockingConfirmationStrategy(
                confirmation_policy=AlwaysAskPolicy(), confirmation_ui=MockUserInterface(ui_result)
            )
        }
    )


class TestConfirmationHook:
    def test_no_tool_calls_is_noop(self, tools):
        messages = [ChatMessage.from_user("hello"), ChatMessage.from_assistant("hi")]
        state = _state_with(messages, tools)
        hook = _confirm_hook(ConfirmationUIResult(action="reject"))

        hook.run(state)

        assert state.get("messages") == messages

    def test_confirm_keeps_tool_call(self, tools):
        tool_call = ToolCall("addition_tool", {"a": 1, "b": 2})
        messages = [ChatMessage.from_user("add"), ChatMessage.from_assistant(tool_calls=[tool_call])]
        state = _state_with(messages, tools)
        # NeverAskPolicy auto-confirms, so the tool call is left untouched for the executor.
        hook = ConfirmationHook(
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            }
        )

        hook.run(state)

        assert state.get("messages")[-1].tool_calls == [tool_call]

    def test_modify_updates_arguments(self, tools):
        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
        ]
        state = _state_with(messages, tools)
        hook = _confirm_hook(ConfirmationUIResult(action="modify", new_tool_params={"a": 10, "b": 20}))

        hook.run(state)

        pending = state.get("messages")[-1].tool_calls
        assert len(pending) == 1
        assert pending[0].arguments == {"a": 10, "b": 20}

    def test_reject_drops_tool_call_and_appends_result(self, tools):
        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
        ]
        state = _state_with(messages, tools)
        hook = _confirm_hook(ConfirmationUIResult(action="reject"))

        hook.run(state)

        new_messages = state.get("messages")
        # A rejection produces a tool-result message and removes the pending call so the executor skips it.
        assert new_messages[-1].tool_call_result is not None
        assert "rejected" in new_messages[-1].tool_call_result.result.lower()

    def test_to_dict(self):
        hook = ConfirmationHook(
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            }
        )
        assert hook.to_dict() == {
            "type": "haystack.human_in_the_loop.hooks.ConfirmationHook",
            "init_parameters": {
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
                }
            },
        }

    def test_from_dict_round_trip(self):
        hook = ConfirmationHook(
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            }
        )
        deserialized = ConfirmationHook.from_dict(hook.to_dict())
        assert deserialized.to_dict() == hook.to_dict()
        strategy = deserialized.confirmation_strategies["addition_tool"]
        assert isinstance(strategy, BlockingConfirmationStrategy)
        assert isinstance(strategy.confirmation_policy, NeverAskPolicy)
        assert isinstance(strategy.confirmation_ui, SimpleConsoleUI)

    def test_tuple_key_round_trips(self):
        hook = ConfirmationHook(
            confirmation_strategies={
                ("addition_tool", "other_tool"): BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            }
        )
        deserialized = ConfirmationHook.from_dict(hook.to_dict())
        assert ("addition_tool", "other_tool") in deserialized.confirmation_strategies


class TestConfirmationHookAsync:
    @pytest.mark.asyncio
    async def test_reject_drops_tool_call_and_appends_result(self, tools):
        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
        ]
        state = _state_with(messages, tools)
        hook = _confirm_hook(ConfirmationUIResult(action="reject"))

        await hook.run_async(state)

        new_messages = state.get("messages")
        assert new_messages[-1].tool_call_result is not None
        assert "rejected" in new_messages[-1].tool_call_result.result.lower()
