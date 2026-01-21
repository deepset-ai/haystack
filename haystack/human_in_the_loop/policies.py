# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.human_in_the_loop import ConfirmationUIResult
from haystack.human_in_the_loop.types import ConfirmationPolicy


class AlwaysAskPolicy(ConfirmationPolicy):
    """Always ask for confirmation."""

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:
        """
        Always ask for confirmation before executing the tool.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns True, indicating confirmation is needed.
        """
        return True


class NeverAskPolicy(ConfirmationPolicy):
    """Never ask for confirmation."""

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:
        """
        Never ask for confirmation, always proceed with tool execution.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns False, indicating no confirmation is needed.
        """
        return False


class AskOncePolicy(ConfirmationPolicy):
    """Ask only once per tool with specific parameters."""

    def __init__(self) -> None:
        self._asked_tools: dict[str, Any] = {}

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:
        """
        Ask for confirmation only once per tool with specific parameters.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: True if confirmation is needed, False if already asked with the same parameters.
        """
        # Don't ask again if we've already asked for this tool with the same parameters
        return not (tool_name in self._asked_tools and self._asked_tools[tool_name] == tool_params)

    def update_after_confirmation(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult,
    ) -> None:
        """
        Store the tool and parameters if the action was "confirm" to avoid asking again.

        This method updates the internal state to remember that the user has already confirmed the execution of the
        tool with the given parameters.

        :param tool_name: The name of the tool that was executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters that were passed to the tool.
        :param confirmation_result: The result from the confirmation UI.
        """
        if confirmation_result.action == "confirm":
            self._asked_tools[tool_name] = tool_params
