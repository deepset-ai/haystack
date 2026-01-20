# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from haystack.components.agents.agent import _ExecutionContext
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ConfirmationUIResult, ToolCall, ToolExecutionDecision
from haystack.human_in_the_loop import (
    AlwaysAskPolicy,
    AskOncePolicy,
    BlockingConfirmationStrategy,
    NeverAskPolicy,
    SimpleConsoleUI,
)
from haystack.human_in_the_loop.strategies import (
    _apply_tool_execution_decisions,
    _run_confirmation_strategies,
    _run_confirmation_strategies_async,
    _update_chat_history,
)
from haystack.tools import Tool, create_tool_from_function


def addition_tool(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def tools() -> list[Tool]:
    tool = create_tool_from_function(
        function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
    )
    return [tool]


@pytest.fixture
def execution_context(tools: list[Tool]) -> _ExecutionContext:
    return _ExecutionContext(
        state=State(schema={"messages": {"type": list[ChatMessage]}}),
        component_visits={"chat_generator": 0, "tool_invoker": 0},
        chat_generator_inputs={},
        tool_invoker_inputs={"tools": tools},
        counter=0,
        skip_chat_generator=False,
    )


class TestBlockingConfirmationStrategy:
    def test_initialization(self):
        strategy = BlockingConfirmationStrategy(confirmation_policy=AskOncePolicy(), confirmation_ui=SimpleConsoleUI())
        assert isinstance(strategy.confirmation_policy, AskOncePolicy)
        assert isinstance(strategy.confirmation_ui, SimpleConsoleUI)

    def test_to_dict(self):
        strategy = BlockingConfirmationStrategy(confirmation_policy=AskOncePolicy(), confirmation_ui=SimpleConsoleUI())
        strategy_dict = strategy.to_dict()
        assert strategy_dict == {
            "type": "haystack.human_in_the_loop.strategies.BlockingConfirmationStrategy",
            "init_parameters": {
                "confirmation_policy": {
                    "type": "haystack.human_in_the_loop.policies.AskOncePolicy",
                    "init_parameters": {},
                },
                "confirmation_ui": {
                    "type": "haystack.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                    "init_parameters": {},
                },
            },
        }

    def test_from_dict(self):
        strategy_dict = {
            "type": "haystack.human_in_the_loop.strategies.BlockingConfirmationStrategy",
            "init_parameters": {
                "confirmation_policy": {
                    "type": "haystack.human_in_the_loop.policies.AskOncePolicy",
                    "init_parameters": {},
                },
                "confirmation_ui": {
                    "type": "haystack.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                    "init_parameters": {},
                },
            },
        }
        strategy = BlockingConfirmationStrategy.from_dict(strategy_dict)
        assert isinstance(strategy, BlockingConfirmationStrategy)
        assert isinstance(strategy.confirmation_policy, AskOncePolicy)
        assert isinstance(strategy.confirmation_ui, SimpleConsoleUI)

    def test_run_confirm(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )

        # Mock the UI to always confirm
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="confirm")

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "value1"}

    def test_run_modify(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )

        # Mock the UI to always modify
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="modify", new_tool_params={"param1": "new_value"})

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "new_value"}
        assert decision.feedback == (
            "The parameters for tool 'test_tool' were updated by the user to:\n{'param1': 'new_value'}"
        )

    def test_run_reject(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )

        # Mock the UI to always reject
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="reject", feedback="Not needed")

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is False
        assert decision.final_tool_params is None
        assert decision.feedback == "Tool execution for 'test_tool' was rejected by the user. With feedback: Not needed"


class TestRunConfirmationStrategies:
    def test_run_confirmation_strategies_no_strategy(self, tools, execution_context):
        teds = _run_confirmation_strategies(
            confirmation_strategies={},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})])
            ],
            execution_context=execution_context,
        )
        assert teds == [
            ToolExecutionDecision(tool_name=tools[0].name, execute=True, final_tool_params={"param1": "value1"})
        ]

    def test_run_confirmation_strategies_with_strategy(self, tools, execution_context):
        teds = _run_confirmation_strategies(
            confirmation_strategies={
                tools[0].name: BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            },
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})])
            ],
            execution_context=execution_context,
        )
        assert teds == [
            ToolExecutionDecision(tool_name=tools[0].name, execute=True, final_tool_params={"param1": "value1"})
        ]

    def test_run_confirmation_strategies_with_existing_teds(self, tools, execution_context):
        exe_context_with_teds = replace(
            execution_context,
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="123", final_tool_params={"param1": "new_value"}
                )
            ],
        )
        teds = _run_confirmation_strategies(
            confirmation_strategies={
                tools[0].name: BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            },
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"}, id="123")])
            ],
            execution_context=exe_context_with_teds,
        )
        assert teds == [
            ToolExecutionDecision(
                tool_name=tools[0].name, execute=True, tool_call_id="123", final_tool_params={"param1": "new_value"}
            )
        ]


class TestApplyToolExecutionDecisions:
    @pytest.fixture
    def assistant_message(self, tools):
        tool_call = ToolCall(tool_name=tools[0].name, arguments={"a": 1, "b": 2}, id="1")
        return ChatMessage.from_assistant(tool_calls=[tool_call])

    def test_reject(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name,
                    execute=False,
                    tool_call_id="1",
                    feedback=(
                        "The tool execution for 'addition_tool' was rejected by the user. With feedback: Not needed"
                    ),
                )
            ],
        )
        assert rejection_messages == [
            assistant_message,
            ChatMessage.from_tool(
                tool_result=(
                    "The tool execution for 'addition_tool' was rejected by the user. With feedback: Not needed"
                ),
                origin=assistant_message.tool_call,
                error=True,
            ),
        ]
        assert new_tool_call_messages == []

    def test_confirm(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="1", final_tool_params={"a": 1, "b": 2}
                )
            ],
        )
        assert rejection_messages == []
        assert new_tool_call_messages == [assistant_message]

    def test_modify(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name,
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'addition_tool' were updated by the user to:\n{'a': 5, 'b': 6}",
                )
            ],
        )
        assert rejection_messages == []
        assert new_tool_call_messages == [
            ChatMessage.from_user(
                "The parameters for tool 'addition_tool' were updated by the user to:\n{'a': 5, 'b': 6}"
            ),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name=tools[0].name, arguments={"a": 5, "b": 6}, id="1")]
            ),
        ]

    def test_two_teds_same_name_no_ids(self):
        message_with_tool_calls = ChatMessage.from_assistant(
            text="I'll extract the information about the people mentioned in the context.",
            # Same tool name with different params but missing IDs
            tool_calls=[
                ToolCall(tool_name="add_database_tool", arguments={"name": "Malte"}),
                ToolCall(tool_name="add_database_tool", arguments={"name": "Milos"}),
            ],
        )
        # This raises a ValueError because tool_call_id is missing and there are multiple tool calls with the same name
        # so we cannot disambiguate which TED applies to which tool call.
        with pytest.raises(
            ValueError,
            match="ToolExecutionDecisions are missing tool_call_id fields and there are multiple tool calls with the "
            "same name",
        ):
            _apply_tool_execution_decisions(
                tool_call_messages=[message_with_tool_calls],
                tool_execution_decisions=[
                    ToolExecutionDecision(
                        tool_name="add_database_tool", execute=True, final_tool_params={"name": "Malte"}
                    ),
                    ToolExecutionDecision(
                        tool_name="add_database_tool", execute=True, final_tool_params={"name": "Milos"}
                    ),
                ],
            )


class TestUpdateChatHistory:
    @pytest.fixture
    def chat_history_one_tool_call(self):
        return [
            ChatMessage.from_user("Hello"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1")]),
        ]

    def test_update_chat_history_rejection(self, chat_history_one_tool_call):
        """Test that the new history includes a tool call result message after a rejection."""
        rejection_messages = [
            ChatMessage.from_assistant(tool_calls=[chat_history_one_tool_call[1].tool_call]),
            ChatMessage.from_tool(
                tool_result="The tool execution for 'tool1' was rejected by the user. With feedback: Not needed",
                origin=chat_history_one_tool_call[1].tool_call,
                error=True,
            ),
        ]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=rejection_messages, tool_call_and_explanation_messages=[]
        )
        assert updated_messages == [chat_history_one_tool_call[0], *rejection_messages]

    def test_update_chat_history_confirm(self, chat_history_one_tool_call):
        """No changes should be made if the tool call was confirmed."""
        tool_call_messages = [ChatMessage.from_assistant(tool_calls=[chat_history_one_tool_call[1].tool_call])]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=[], tool_call_and_explanation_messages=tool_call_messages
        )
        assert updated_messages == chat_history_one_tool_call

    def test_update_chat_history_modify(self, chat_history_one_tool_call):
        """Test that the new history includes a user message and updated tool call after a modification."""
        tool_call_messages = [
            ChatMessage.from_user(
                "The parameters for tool 'tool1' were updated by the user to:\n{'param': 'new_value'}"
            ),
            ChatMessage.from_assistant(tool_calls=[ToolCall("tool1", {"param": "new_value"}, id="1")]),
        ]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=[], tool_call_and_explanation_messages=tool_call_messages
        )
        assert updated_messages == [chat_history_one_tool_call[0], *tool_call_messages]

    def test_update_chat_history_modify_two_tool_calls(self):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1"), ToolCall("tool2", {"a": 3, "b": 4}, id="2")]
        )
        chat_history = [ChatMessage.from_user("What is 1 + 2? and 3 + 4?"), tool_call_message]
        rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[tool_call_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name="tool1",
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}",
                ),
                ToolExecutionDecision(
                    tool_name="tool2",
                    execute=True,
                    tool_call_id="2",
                    final_tool_params={"a": 7, "b": 8},
                    feedback="The parameters for tool 'tool2' were updated by the user to:\n{'a': 7, 'b': 8}'",
                ),
            ],
        )
        updated_messages = _update_chat_history(
            chat_history,
            rejection_messages=rejection_messages,
            tool_call_and_explanation_messages=modified_tool_call_messages,
        )
        assert updated_messages == [
            chat_history[0],
            ChatMessage.from_user("The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}"),
            ChatMessage.from_user("The parameters for tool 'tool2' were updated by the user to:\n{'a': 7, 'b': 8}'"),
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(tool_name="tool1", arguments={"a": 5, "b": 6}, id="1", extra=None),
                    ToolCall(tool_name="tool2", arguments={"a": 7, "b": 8}, id="2", extra=None),
                ]
            ),
        ]

    def test_update_chat_history_two_tool_calls_modify_and_reject(self):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1"), ToolCall("tool2", {"a": 3, "b": 4}, id="2")]
        )
        chat_history = [ChatMessage.from_user("What is 1 + 2? and 3 + 4?"), tool_call_message]
        rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[tool_call_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name="tool1",
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}",
                ),
                ToolExecutionDecision(
                    tool_name="tool2",
                    execute=False,
                    tool_call_id="2",
                    feedback="The tool execution for 'tool2' was rejected by the user. With feedback: Not needed",
                ),
            ],
        )
        updated_messages = _update_chat_history(
            chat_history,
            rejection_messages=rejection_messages,
            tool_call_and_explanation_messages=modified_tool_call_messages,
        )
        assert updated_messages == [
            chat_history[0],
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="tool2", arguments={"a": 3, "b": 4}, id="2", extra=None)]
            ),
            ChatMessage.from_tool(
                tool_result="The tool execution for 'tool2' was rejected by the user. With feedback: Not needed",
                origin=ToolCall(tool_name="tool2", arguments={"a": 3, "b": 4}, id="2", extra=None),
                error=True,
            ),
            ChatMessage.from_user("The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="tool1", arguments={"a": 5, "b": 6}, id="1", extra=None)]
            ),
        ]


class ConfirmationStrategyContextCapturingStrategy:
    def __init__(self):
        self.captured_confirmation_strategy_context: dict[str, Any] | None = None

    def run(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        self.captured_confirmation_strategy_context = confirmation_strategy_context
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    async def run_async(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        self.captured_confirmation_strategy_context = confirmation_strategy_context
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "test.RunContextCapturingStrategy", "init_parameters": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationStrategyContextCapturingStrategy":
        return cls()


class TrueAsyncConfirmationStrategy:
    """
    A confirmation strategy with truly async behavior for testing purposes.

    This strategy simulates an async operation (like waiting for a WebSocket response)
    by using asyncio.sleep. It demonstrates how custom strategies can implement
    non-blocking async confirmation flows.
    """

    def __init__(self, delay: float = 0.01, decision: str = "confirm"):
        self.delay = delay
        self.decision = decision
        self.async_was_called = False
        self.sync_was_called = False

    def run(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """Sync version - should NOT be called when run_async is available."""
        self.sync_was_called = True
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    async def run_async(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """Truly async version that simulates waiting for external confirmation."""
        self.async_was_called = True

        # Simulate async operation (e.g., waiting for WebSocket response)
        await asyncio.sleep(self.delay)

        if self.decision == "reject":
            return ToolExecutionDecision(
                tool_name=tool_name,
                execute=False,
                tool_call_id=tool_call_id,
                feedback=f"Tool '{tool_name}' was rejected asynchronously.",
            )
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "test.TrueAsyncConfirmationStrategy", "init_parameters": {"delay": self.delay}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrueAsyncConfirmationStrategy":
        return cls(**data.get("init_parameters", {}))


class TestRunContext:
    def test_confirmation_strategy_context_passed_to_strategy(self, tools):
        confirmation_strategy_context = {"event_queue": "mock_queue", "redis_client": "mock_redis"}
        execution_context = _ExecutionContext(
            state=State(schema={"messages": {"type": list[ChatMessage]}}),
            component_visits={"chat_generator": 0, "tool_invoker": 0},
            chat_generator_inputs={},
            tool_invoker_inputs={"tools": tools},
            counter=0,
            skip_chat_generator=False,
            confirmation_strategy_context=confirmation_strategy_context,
        )

        capturing_strategy = ConfirmationStrategyContextCapturingStrategy()
        teds = _run_confirmation_strategies(
            confirmation_strategies={tools[0].name: capturing_strategy},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})])
            ],
            execution_context=execution_context,
        )

        # Verify the strategy received the confirmation_strategy_context directly
        assert capturing_strategy.captured_confirmation_strategy_context is not None
        assert capturing_strategy.captured_confirmation_strategy_context == confirmation_strategy_context
        assert len(teds) == 1
        assert teds[0].execute is True

    @pytest.mark.asyncio
    async def test_confirmation_strategy_context_passed_to_strategy_async(self, tools):
        confirmation_strategy_context = {"websocket": "mock_websocket", "request_id": "12345"}
        execution_context = _ExecutionContext(
            state=State(schema={"messages": {"type": list[ChatMessage]}}),
            component_visits={"chat_generator": 0, "tool_invoker": 0},
            chat_generator_inputs={},
            tool_invoker_inputs={"tools": tools},
            counter=0,
            skip_chat_generator=False,
            confirmation_strategy_context=confirmation_strategy_context,
        )

        capturing_strategy = ConfirmationStrategyContextCapturingStrategy()
        teds = await _run_confirmation_strategies_async(
            confirmation_strategies={tools[0].name: capturing_strategy},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})])
            ],
            execution_context=execution_context,
        )

        # Verify the strategy received the confirmation_strategy_context directly
        assert capturing_strategy.captured_confirmation_strategy_context is not None
        assert capturing_strategy.captured_confirmation_strategy_context == confirmation_strategy_context
        assert len(teds) == 1
        assert teds[0].execute is True


class TestAsyncConfirmationStrategies:
    @pytest.mark.asyncio
    async def test_async_strategy_confirm(self):
        strategy = TrueAsyncConfirmationStrategy(delay=0.01, decision="confirm")

        decision = await strategy.run_async(
            tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"}
        )

        assert strategy.async_was_called is True
        assert strategy.sync_was_called is False
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_async_strategy_reject(self):
        strategy = TrueAsyncConfirmationStrategy(delay=0.01, decision="reject")

        decision = await strategy.run_async(
            tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"}
        )

        assert strategy.async_was_called is True
        assert decision.execute is False
        assert decision.feedback is not None and "rejected asynchronously" in decision.feedback

    @pytest.mark.asyncio
    async def test_async_strategy_used_by_run_confirmation_strategies_async(self, tools, execution_context):
        strategy = TrueAsyncConfirmationStrategy(delay=0.01, decision="confirm")

        teds = await _run_confirmation_strategies_async(
            confirmation_strategies={tools[0].name: strategy},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})])
            ],
            execution_context=execution_context,
        )

        # Verify that only the async method was called
        assert strategy.async_was_called is True
        assert strategy.sync_was_called is False
        assert len(teds) == 1
        assert teds[0].execute is True

    @pytest.mark.asyncio
    async def test_blocking_strategy_run_async(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )

        # Mock the UI to always confirm
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="confirm")

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = await strategy.run_async(
            tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"}
        )
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_run_confirmation_strategies_async_no_strategy(self, tools, execution_context):
        teds = await _run_confirmation_strategies_async(
            confirmation_strategies={},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})])
            ],
            execution_context=execution_context,
        )
        assert len(teds) == 1
        assert teds[0].tool_name == tools[0].name
        assert teds[0].execute is True
        assert teds[0].final_tool_params == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_run_confirmation_strategies_async_with_strategy(self, tools, execution_context):
        teds = await _run_confirmation_strategies_async(
            confirmation_strategies={
                tools[0].name: BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            },
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})])
            ],
            execution_context=execution_context,
        )
        assert len(teds) == 1
        assert teds[0].tool_name == tools[0].name
        assert teds[0].execute is True

    @pytest.mark.asyncio
    async def test_run_confirmation_strategies_async_with_existing_teds(self, tools, execution_context):
        exe_context_with_teds = replace(
            execution_context,
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="123", final_tool_params={"a": 5, "b": 6}
                )
            ],
        )
        teds = await _run_confirmation_strategies_async(
            confirmation_strategies={
                tools[0].name: BlockingConfirmationStrategy(
                    confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
                )
            },
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2}, id="123")])
            ],
            execution_context=exe_context_with_teds,
        )
        # Should use the existing TED, not create a new one
        assert len(teds) == 1
        assert teds[0].tool_call_id == "123"
        assert teds[0].final_tool_params == {"a": 5, "b": 6}
