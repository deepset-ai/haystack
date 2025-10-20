# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Union


@dataclass(frozen=True)
class Breakpoint:
    """
    A dataclass to hold a breakpoint for a component.

    :param component_name: The name of the component where the breakpoint is set.
    :param visit_count: The number of times the component must be visited before the breakpoint is triggered.
    :param snapshot_file_path: Optional path to store a snapshot of the pipeline when the breakpoint is hit.
        This is useful for debugging purposes, allowing you to inspect the state of the pipeline at the time of the
        breakpoint and to resume execution from that point.
    """

    component_name: str
    visit_count: int = 0
    snapshot_file_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Breakpoint to a dictionary representation.

        :return: A dictionary containing the component name, visit count, and debug path.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Breakpoint":
        """
        Populate the Breakpoint from a dictionary representation.

        :param data: A dictionary containing the component name, visit count, and debug path.
        :return: An instance of Breakpoint.
        """
        return cls(**data)


@dataclass(frozen=True)
class ToolBreakpoint(Breakpoint):
    """
    A dataclass representing a breakpoint specific to tools used within an Agent component.

    Inherits from Breakpoint and adds the ability to target individual tools. If `tool_name` is None,
    the breakpoint applies to all tools within the Agent component.

    :param tool_name: The name of the tool to target within the Agent component. If None, applies to all tools.
    """

    tool_name: Optional[str] = None

    def __str__(self) -> str:
        tool_str = f", tool_name={self.tool_name}" if self.tool_name else ", tool_name=ALL_TOOLS"
        return f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}{tool_str})"


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AgentBreakpoint to a dictionary representation.

        :return: A dictionary containing the agent name and the breakpoint details.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentBreakpoint":
        """
        Populate the AgentBreakpoint from a dictionary representation.

        :param data: A dictionary containing the agent name and the breakpoint details.
        :return: An instance of AgentBreakpoint.
        """
        break_point_data = data["break_point"]
        break_point: Union[Breakpoint, ToolBreakpoint]
        if "tool_name" in break_point_data:
            break_point = ToolBreakpoint(**break_point_data)
        else:
            break_point = Breakpoint(**break_point_data)
        return cls(agent_name=data["agent_name"], break_point=break_point)


@dataclass
class AgentSnapshot:
    component_inputs: dict[str, Any]
    component_visits: dict[str, int]
    break_point: AgentBreakpoint
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AgentSnapshot to a dictionary representation.

        :return: A dictionary containing the agent state, timestamp, and breakpoint.
        """
        return {
            "component_inputs": self.component_inputs,
            "component_visits": self.component_visits,
            "break_point": self.break_point.to_dict(),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSnapshot":
        """
        Populate the AgentSnapshot from a dictionary representation.

        :param data: A dictionary containing the agent state, timestamp, and breakpoint.
        :return: An instance of AgentSnapshot.
        """
        return cls(
            component_inputs=data["component_inputs"],
            component_visits=data["component_visits"],
            break_point=AgentBreakpoint.from_dict(data["break_point"]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
        )


@dataclass
class PipelineState:
    """
    A dataclass to hold the state of the pipeline at a specific point in time.

    :param component_visits: A dictionary mapping component names to their visit counts.
    :param inputs: The inputs processed by the pipeline at the time of the snapshot.
    :param pipeline_outputs: Dictionary containing the final outputs of the pipeline up to the breakpoint.
    """

    inputs: dict[str, Any]
    component_visits: dict[str, int]
    pipeline_outputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PipelineState to a dictionary representation.

        :return: A dictionary containing the inputs, component visits,
                and pipeline outputs.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        """
        Populate the PipelineState from a dictionary representation.

        :param data: A dictionary containing the inputs, component visits,
                    and pipeline outputs.
        :return: An instance of PipelineState.
        """
        return cls(**data)


@dataclass
class PipelineSnapshot:
    """
    A dataclass to hold a snapshot of the pipeline at a specific point in time.

    :param original_input_data: The original input data provided to the pipeline.
    :param ordered_component_names: A list of component names in the order they were visited.
    :param pipeline_state: The state of the pipeline at the time of the snapshot.
    :param break_point: The breakpoint that triggered the snapshot.
    :param agent_snapshot: Optional agent snapshot if the breakpoint is an agent breakpoint.
    :param timestamp: A timestamp indicating when the snapshot was taken.
    :param include_outputs_from: Set of component names whose outputs should be included in the pipeline results.
    """

    original_input_data: dict[str, Any]
    ordered_component_names: list[str]
    pipeline_state: PipelineState
    break_point: Union[AgentBreakpoint, Breakpoint]
    agent_snapshot: Optional[AgentSnapshot] = None
    timestamp: Optional[datetime] = None
    include_outputs_from: set[str] = field(default_factory=set)

    def __post_init__(self):
        # Validate consistency between component_visits and ordered_component_names
        components_in_state = set(self.pipeline_state.component_visits.keys())
        components_in_order = set(self.ordered_component_names)

        if components_in_state != components_in_order:
            raise ValueError(
                f"Inconsistent state: components in PipelineState.component_visits {components_in_state} "
                f"do not match components in PipelineSnapshot.ordered_component_names {components_in_order}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PipelineSnapshot to a dictionary representation.

        :return: A dictionary containing the pipeline state, timestamp, breakpoint, agent snapshot, original input data,
                 ordered component names, include_outputs_from, and pipeline outputs.
        """
        data = {
            "pipeline_state": self.pipeline_state.to_dict(),
            "break_point": self.break_point.to_dict(),
            "agent_snapshot": self.agent_snapshot.to_dict() if self.agent_snapshot else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "original_input_data": self.original_input_data,
            "ordered_component_names": self.ordered_component_names,
            "include_outputs_from": list(self.include_outputs_from),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineSnapshot":
        """
        Populate the PipelineSnapshot from a dictionary representation.

        :param data: A dictionary containing the pipeline state, timestamp, breakpoint, agent snapshot, original input
                     data, ordered component names, include_outputs_from, and pipeline outputs.
        """
        # Convert include_outputs_from list back to set for serialization
        include_outputs_from = set(data.get("include_outputs_from", []))

        return cls(
            pipeline_state=PipelineState.from_dict(data=data["pipeline_state"]),
            break_point=(
                AgentBreakpoint.from_dict(data=data["break_point"])
                if "agent_name" in data["break_point"]
                else Breakpoint.from_dict(data=data["break_point"])
            ),
            agent_snapshot=AgentSnapshot.from_dict(data["agent_snapshot"]) if data.get("agent_snapshot") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            original_input_data=data.get("original_input_data", {}),
            ordered_component_names=data.get("ordered_component_names", []),
            include_outputs_from=include_outputs_from,
        )
