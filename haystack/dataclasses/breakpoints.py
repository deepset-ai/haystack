# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Breakpoint:
    """
    A dataclass to hold a breakpoint for a component.
    """

    component_name: str
    visit_count: int = 0

    def __hash__(self):
        return hash((self.component_name, self.visit_count))

    def __eq__(self, other):
        if not isinstance(other, Breakpoint):
            return False
        return self.component_name == other.component_name and self.visit_count == other.visit_count

    def __str__(self):
        return f"Breakpoint(component_name={self.component_name}, visit_count={self.visit_count})"

    def __repr__(self):
        return self.__str__()


@dataclass
class ToolBreakpoint(Breakpoint):
    """
    A dataclass to hold a breakpoint that can be used to debug a Tool.

    If tool_name is None, it means that the breakpoint is for every tool in the component.
    Otherwise, it means that the breakpoint is for the tool with the given name.
    """

    tool_name: Optional[str] = None

    def __hash__(self):
        return hash((self.component_name, self.visit_count, self.tool_name))

    def __eq__(self, other):
        if not isinstance(other, ToolBreakpoint):
            return False
        return super().__eq__(other) and self.tool_name == other.tool_name

    def __str__(self):
        if self.tool_name:
            return (
                f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}, "
                f"tool_name={self.tool_name})"
            )
        else:
            return (
                f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}, "
                f"tool_name=ALL_TOOLS)"
            )

    def __repr__(self):
        return self.__str__()


@dataclass
class AgentBreakpoint:
    """
    A dataclass to hold a breakpoint that can be used to debug an Agent.
    """

    break_point: Union[Breakpoint, ToolBreakpoint]
    agent_name: str = ""

    def __init__(self, agent_name: str, break_point: Union[Breakpoint, ToolBreakpoint]):
        if not isinstance(break_point, ToolBreakpoint) and break_point.component_name != "chat_generator":
            raise ValueError(
                "The break_point must be a Breakpoint that has the component_name "
                "'chat_generator' or be a ToolBreakpoint."
            )

        if not break_point:
            raise ValueError("A Breakpoint must be provided.")

        self.agent_name = agent_name

        if (
            isinstance(break_point, ToolBreakpoint)
            or isinstance(break_point, Breakpoint)
            and not isinstance(break_point, ToolBreakpoint)
        ):
            self.break_point = break_point
        else:
            raise ValueError("The breakpoint must be either Breakpoint or ToolBreakpoint.")
