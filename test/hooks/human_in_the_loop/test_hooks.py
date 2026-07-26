# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any

import pytest

from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import merge_lists, replace_values
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    ConfirmationUIResult,
    NeverAskPolicy,
    SimpleConsoleUI,
    ToolExecutionDecision,
)
from haystack.hooks.human_in_the_loop.types import ConfirmationUI
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


def _confirm_strat(ui_result: ConfirmationUIResult) -> BlockingConfirmationStrategy:
    return BlockingConfirmationStrategy(
        confirmation_policy=AlwaysAskPolicy(), confirmation_ui=MockUserInterface(ui_result)
    )


def _confirm_hook(ui_result: ConfirmationUIResult, tool_name: str = "addition_tool") -> ConfirmationHook:
    return ConfirmationHook(confirmation_strategies={tool_name: _confirm_strat(ui_result)})


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
        # The original call is dropped; an explanation user message precedes the rebuilt call with the new arguments.
        explanation = "The parameters for tool 'addition_tool' were updated by the user to:\n{'a': 10, 'b': 20}"
        assert state.get("messages") == [
            ChatMessage.from_user("add"),
            ChatMessage.from_user(explanation),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 10, "b": 20})]),
        ]

    def test_reject_drops_tool_call_and_appends_result(self, tools):
        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
        ]
        state = _state_with(messages, tools)
        hook = _confirm_hook(ConfirmationUIResult(action="reject"))
        hook.run(state)
        # The call is answered by an error tool result, so it is resolved and the executor skips it (no pending call).
        assert state.get("messages") == [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
            ChatMessage.from_tool(
                tool_result="Tool execution for 'addition_tool' was rejected by the user.",
                origin=ToolCall("addition_tool", {"a": 1, "b": 2}),
                error=True,
            ),
        ]

    def test_hook_context_is_not_deepcopied(self, tools):
        # A non-copyable resource in hook_context (e.g. a lock, WebSocket, or client) must reach the strategy
        # unchanged. Reading via state.get would deepcopy and raise; the hook reads via state.data instead.
        lock = threading.Lock()
        received = {}

        class CapturingStrategy:
            def run(
                self, *, tool_name, tool_description, tool_params, tool_call_id=None, confirmation_strategy_context=None
            ):
                received["context"] = confirmation_strategy_context
                return ToolExecutionDecision(
                    tool_name=tool_name, tool_call_id=tool_call_id, execute=True, final_tool_params=tool_params
                )

        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
        ]
        state = _state_with(messages, tools)
        state.data["hook_context"] = {"resource": lock}
        ConfirmationHook(confirmation_strategies={"addition_tool": CapturingStrategy()}).run(state)  # type: ignore[dict-item]
        # Passed through by identity: the non-copyable resource is neither copied nor does the read raise.
        assert received["context"]["resource"] is lock

    def test_to_dict(self):
        hook = ConfirmationHook(
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            }
        )
        assert hook.to_dict() == {
            "type": "haystack.hooks.human_in_the_loop.hooks.ConfirmationHook",
            "init_parameters": {
                "confirmation_strategies": {
                    "addition_tool": {
                        "type": "haystack.hooks.human_in_the_loop.strategies.BlockingConfirmationStrategy",
                        "init_parameters": {
                            "confirmation_policy": {
                                "type": "haystack.hooks.human_in_the_loop.policies.NeverAskPolicy",
                                "init_parameters": {},
                            },
                            "confirmation_ui": {
                                "type": "haystack.hooks.human_in_the_loop.user_interfaces.SimpleConsoleUI",
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


class TestConfirmationHookWildcard:
    def test_wildcard_applies_to_any_tool(self, tools):
        # No entry for "addition_tool"; the "*" wildcard covers it.
        messages = [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall(id="tc-1", tool_name="addition_tool", arguments={})]),
        ]
        state = _state_with(messages, tools)
        _confirm_hook(ConfirmationUIResult(action="reject"), tool_name="*").run(state)

        new_messages = state.get("messages")
        assert new_messages[-1].tool_call_result is not None
        assert "rejected" in new_messages[-1].tool_call_result.result.lower()

    def test_specific_key_beats_wildcard(self):
        add_tool = create_tool_from_function(
            function=addition_tool, name="addition_tool", description="Adds two integers."
        )
        other_tool = create_tool_from_function(function=addition_tool, name="other_tool", description="Another tool.")
        messages = [
            ChatMessage.from_user("go"),
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(id="tc-add", tool_name="addition_tool", arguments={"a": 1, "b": 2}),
                    ToolCall(id="tc-other", tool_name="other_tool", arguments={"a": 3, "b": 4}),
                ]
            ),
        ]
        state = _state_with(messages, [add_tool, other_tool])
        # addition_tool has its own (confirm) entry; other_tool falls through to the "*" (reject) wildcard.
        hook = ConfirmationHook(
            confirmation_strategies={
                "addition_tool": _confirm_strat(ConfirmationUIResult(action="confirm")),
                "*": _confirm_strat(ConfirmationUIResult(action="reject")),
            }
        )
        hook.run(state)

        new_messages = state.get("messages")
        # addition_tool is confirmed, so it stays pending on the last message; other_tool is rejected.
        assert [tc.tool_name for tc in new_messages[-1].tool_calls] == ["addition_tool"]
        rejected = [m for m in new_messages if m.tool_call_result is not None]
        assert [m.tool_call_result.origin.tool_name for m in rejected] == ["other_tool"]
        assert rejected[0].tool_call_result.error is True


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
        # run_async produces the same transcript as run: the call answered by an error tool result, no pending call.
        assert state.get("messages") == [
            ChatMessage.from_user("add"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("addition_tool", {"a": 1, "b": 2})]),
            ChatMessage.from_tool(
                tool_result="Tool execution for 'addition_tool' was rejected by the user.",
                origin=ToolCall("addition_tool", {"a": 1, "b": 2}),
                error=True,
            ),
        ]


def _echo(x: int) -> dict[str, int]:
    return {"echoed": x}


def _echo_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Echo tool {name}.",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        function=_echo,
    )


def _multi_tool_hook(decisions: dict[str, ConfirmationUIResult]) -> ConfirmationHook:
    return ConfirmationHook(
        confirmation_strategies={name: _confirm_strat(ui_result) for name, ui_result in decisions.items()}
    )


def _assert_pending_calls_only_in_last_message(messages: list[ChatMessage]) -> None:
    """
    Assert the invariant the Agent loop relies on: after confirmation, every tool call that is still pending (an
    assistant tool call with no matching tool result later in the history) lives in the last message. The loop reads
    the calls to execute from the last message alone, so any leftover pending call elsewhere would silently never run.
    """
    resolved_ids = {m.tool_call_result.origin.id for m in messages if m.tool_call_result is not None}
    for index, message in enumerate(messages):
        pending = [tc for tc in (message.tool_calls or []) if tc.id not in resolved_ids]
        if pending:
            assert index == len(messages) - 1, f"pending calls in non-last message at index {index}: {pending}"


class TestConfirmationHookPendingInvariant:
    """The pending tool calls left after the hook runs must always sit in the last message of the chat history."""

    def _run(self, decisions: dict[str, ConfirmationUIResult]) -> list[ChatMessage]:
        names = list(decisions)
        tools = [_echo_tool(name) for name in names]
        tool_calls = [ToolCall(id=f"tc-{name}", tool_name=name, arguments={"x": i}) for i, name in enumerate(names)]
        messages = [ChatMessage.from_user("go"), ChatMessage.from_assistant(tool_calls=tool_calls)]
        state = _state_with(messages, tools)
        _multi_tool_hook(decisions).run(state)
        return state.get("messages")

    def test_all_confirmed(self):
        messages = self._run(
            {"foo": ConfirmationUIResult(action="confirm"), "bar": ConfirmationUIResult(action="confirm")}
        )
        _assert_pending_calls_only_in_last_message(messages)
        # Both calls survive, untouched, on the (single) last assistant message.
        assert {tc.tool_name for tc in messages[-1].tool_calls} == {"foo", "bar"}

    def test_all_rejected(self):
        messages = self._run(
            {"foo": ConfirmationUIResult(action="reject"), "bar": ConfirmationUIResult(action="reject")}
        )
        _assert_pending_calls_only_in_last_message(messages)
        # Nothing left to run: the last message is a tool result, not a pending tool call.
        assert not messages[-1].tool_calls

    def test_mixed_reject_and_confirm(self):
        messages = self._run(
            {"foo": ConfirmationUIResult(action="reject"), "bar": ConfirmationUIResult(action="confirm")}
        )
        _assert_pending_calls_only_in_last_message(messages)
        # The rejected call is resolved earlier; only the confirmed survivor remains pending, on the last message.
        assert [tc.tool_name for tc in messages[-1].tool_calls] == ["bar"]

    def test_mixed_reject_and_modify(self):
        messages = self._run(
            {
                "foo": ConfirmationUIResult(action="reject"),
                "bar": ConfirmationUIResult(action="modify", new_tool_params={"x": 99}),
            }
        )
        _assert_pending_calls_only_in_last_message(messages)
        assert [tc.tool_name for tc in messages[-1].tool_calls] == ["bar"]
        assert messages[-1].tool_calls[0].arguments == {"x": 99}
