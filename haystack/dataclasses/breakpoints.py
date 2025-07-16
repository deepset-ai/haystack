# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Breakpoint:
    """
    A dataclass to hold a breakpoint for a component.

    :param component_name: The name of the component where the breakpoint is set.
    :param visit_count: The number of times the component must be visited before the breakpoint is triggered.
    :param debug_path: Optional path to store the state of the pipeline when the breakpoint is hit.
        This is useful for debugging purposes, allowing you to inspect the state of the pipeline at the time of the
        breakpoint and to resume execution from that point.
    """

    component_name: str
    visit_count: int = 0
    debug_path: Optional[str] = None


@dataclass(frozen=True)
class ToolBreakpoint(Breakpoint):
    """
    A dataclass representing a breakpoint specific to tools used within an Agent component.

    Inherits from Breakpoint and adds the ability to target individual tools. If `tool_name` is None,
    the breakpoint applies to all tools within the Agent component.

    :param tool_name: The name of the tool to target within the Agent component. If None, applies to all tools.
    """

    tool_name: Optional[str] = None

    def __str__(self):
        tool_str = f", tool_name={self.tool_name}" if self.tool_name else ", tool_name=ALL_TOOLS"
        return f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}{tool_str})"


@dataclass
class AgentBreakpoint:
    """
    A dataclass representing a breakpoint tied to an Agent’s execution.

    This allows for debugging either a specific component (e.g., the chat generator) or a tool used by the agent.
    It enforces constraints on which component names are valid for each breakpoint type.

    :param agent_name: The name of the agent component in a pipeline where the breakpoint is set.
    :param break_point: An instance of Breakpoint or ToolBreakpoint indicating where to break execution.

    :raises ValueError: If the component_name is invalid for the given breakpoint type:
        - Breakpoint must have component_name='chat_generator'.
        - ToolBreakpoint must have component_name='tool_invoker'.
    """

    agent_name: str
    break_point: Union[Breakpoint, ToolBreakpoint]

    def __post_init__(self):
        if (
            isinstance(self.break_point, Breakpoint) and not isinstance(self.break_point, ToolBreakpoint)
        ) and self.break_point.component_name != "chat_generator":
            raise ValueError("If the break_point is a Breakpoint, it must have the component_name 'chat_generator'.")

        if isinstance(self.break_point, ToolBreakpoint) and self.break_point.component_name != "tool_invoker":
            raise ValueError("If the break_point is a ToolBreakpoint, it must have the component_name 'tool_invoker'.")
