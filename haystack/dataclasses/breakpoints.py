# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Breakpoint:
    """
    A dataclass to hold a breakpoint for a component.
    """

    component_name: str
    visit_count: int = 0


@dataclass(frozen=True)
class ToolBreakpoint(Breakpoint):
    """
    A dataclass to hold a breakpoint that can be used to debug a Tool.

    If tool_name is None, it means that the breakpoint is for every tool in the component.
    Otherwise, it means that the breakpoint is for the tool with the given name.
    """

    tool_name: Optional[str] = None

    def __str__(self):
        tool_str = f", tool_name={self.tool_name}" if self.tool_name else ", tool_name=ALL_TOOLS"
        return f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}{tool_str})"


@dataclass
class AgentBreakpoint:
    """
    A dataclass to hold a breakpoint that can be used to debug an Agent.
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
