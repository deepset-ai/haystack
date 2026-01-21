# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.human_in_the_loop.dataclasses import ConfirmationUIResult, ToolExecutionDecision

# Ellipsis are needed to define the Protocol but pylint complains. See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis


class ConfirmationUI(Protocol):
    """Base class for confirmation UIs."""

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        """Get user confirmation for tool execution."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the UI to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI":
        """Deserialize the ConfirmationUI from a dictionary."""
        return default_from_dict(cls, data)


class ConfirmationPolicy(Protocol):
    """Base class for confirmation policies."""

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:
        """Determine whether to ask for confirmation."""
        ...

    def update_after_confirmation(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult,
    ) -> None:
        """Update the policy based on the confirmation UI result."""
        return

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy":
        """Deserialize the policy from a dictionary."""
        return default_from_dict(cls, data)


class ConfirmationStrategy(Protocol):
    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Run the confirmation strategy for a given tool and its parameters.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :param tool_call_id: Optional unique identifier for the tool call. This can be used to track and correlate
            the decision with a specific tool invocation.
        :param confirmation_strategy_context: Optional context dictionary for passing request-scoped resources
            (e.g., WebSocket connections, async queues) in web/server environments.

        :returns:
            The result of the confirmation strategy (e.g., tool output, rejection message, etc.).
        """
        ...

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Async version of run. Run the confirmation strategy for a given tool and its parameters.

        Default implementation calls the sync run() method. Override for true async behavior.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :param tool_call_id: Optional unique identifier for the tool call. This can be used to track and correlate
            the decision with a specific tool invocation.
        :param confirmation_strategy_context: Optional context dictionary for passing request-scoped resources
            (e.g., WebSocket connections, async queues) in web/server environments.

        :returns:
            The result of the confirmation strategy (e.g., tool output, rejection message, etc.).
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationStrategy":
        """Deserialize the strategy from a dictionary."""
        ...
