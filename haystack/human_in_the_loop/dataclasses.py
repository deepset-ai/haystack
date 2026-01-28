# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConfirmationUIResult:
    """
    Result of the confirmation UI interaction.

    :param action:
        The action taken by the user such as "confirm", "reject", or "modify".
        This action type is not enforced to allow for custom actions to be implemented.
    :param feedback:
        Optional feedback message from the user. For example, if the user rejects the tool execution,
        they might provide a reason for the rejection.
    :param new_tool_params:
        Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
        they can provide a new set of parameters here.
    """

    action: str  # "confirm", "reject", "modify"
    feedback: str | None = None
    new_tool_params: dict[str, Any] | None = None


@dataclass
class ToolExecutionDecision:
    """
    Decision made regarding tool execution.

    :param tool_name:
        The name of the tool to be executed.
    :param execute:
        A boolean indicating whether to execute the tool with the provided parameters.
    :param tool_call_id:
        Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
        specific tool invocation.
    :param feedback:
        Optional feedback message.
        For example, if the tool execution is rejected, this can contain the reason. Or if the tool parameters were
        modified, this can contain the modification details.
    :param final_tool_params:
        Optional final parameters for the tool if execution is confirmed or modified.
    """

    tool_name: str
    execute: bool
    tool_call_id: str | None = None
    feedback: str | None = None
    final_tool_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ToolExecutionDecision to a dictionary representation.

        :return: A dictionary containing the tool execution decision details.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionDecision":
        """
        Populate the ToolExecutionDecision from a dictionary representation.

        :param data: A dictionary containing the tool execution decision details.
        :return: An instance of ToolExecutionDecision.
        """
        return cls(**data)
