# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.human_in_the_loop.strategies import (
    _deserialize_confirmation_strategies,
    _process_confirmation_strategies,
    _process_confirmation_strategies_async,
    _serialize_confirmation_strategies,
)
from haystack.human_in_the_loop.types import ConfirmationStrategy


class ConfirmationHook:
    """
    A `before_tool` Agent hook that applies Human-in-the-Loop confirmation strategies to pending tool calls.

    Register it on an `Agent` to confirm, modify, or reject tool calls before they run:

    ```python
    from haystack.components.agents import Agent
    from haystack.human_in_the_loop import (
        BlockingConfirmationStrategy,
        ConfirmationHook,
        NeverAskPolicy,
        SimpleConsoleUI,
    )

    hook = ConfirmationHook(
        confirmation_strategies={
            "my_tool": BlockingConfirmationStrategy(
                confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
            )
        }
    )
    agent = Agent(chat_generator=..., tools=[...], hooks={"before_tool": [hook]})
    ```

    Request-scoped resources for the strategies (e.g. a WebSocket or queue) are passed per run via the Agent's
    `hook_context` argument (`agent.run(messages=[...], hook_context={...})`) and read by the hook with
    `state.get("hook_context")`.
    """

    def __init__(self, confirmation_strategies: dict[str | tuple[str, ...], ConfirmationStrategy]) -> None:
        """
        :param confirmation_strategies: Mapping of tool name (or a tuple of tool names) to its `ConfirmationStrategy`.
        """
        self.confirmation_strategies = confirmation_strategies

    def run(self, state: State) -> None:
        """
        Confirm the pending tool calls, rewriting the `messages` in `state` to reflect modifications and rejections.

        :param state: The Agent's live `State`. Reads the available tools (`state.get("tools")`) and the per-run
            context (`state.get("hook_context")`), and the pending tool calls from the last message; writes the
            updated conversation back to `messages`.
        """
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        _, new_chat_history = _process_confirmation_strategies(
            confirmation_strategies=self.confirmation_strategies,
            messages_with_tool_calls=[messages[-1]],
            tools=state.get("tools") or [],
            state=state,
            confirmation_strategy_context=state.get("hook_context"),
        )
        state.set("messages", new_chat_history, handler_override=replace_values)

    async def run_async(self, state: State) -> None:
        """Async version of `run`."""
        messages = state.data.get("messages") or []
        if not messages or not messages[-1].tool_calls:
            return
        _, new_chat_history = await _process_confirmation_strategies_async(
            confirmation_strategies=self.confirmation_strategies,
            messages_with_tool_calls=[messages[-1]],
            tools=state.get("tools") or [],
            state=state,
            confirmation_strategy_context=state.get("hook_context"),
        )
        state.set("messages", new_chat_history, handler_override=replace_values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hook, including its confirmation strategies (tuple keys become JSON-array strings)."""
        return default_to_dict(
            self, confirmation_strategies=_serialize_confirmation_strategies(self.confirmation_strategies)
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationHook":
        """Deserialize the hook, reconstructing its confirmation strategies."""
        init_params = data.get("init_parameters", {})
        if init_params.get("confirmation_strategies") is not None:
            init_params["confirmation_strategies"] = _deserialize_confirmation_strategies(
                init_params["confirmation_strategies"]
            )
        return default_from_dict(cls, data)
