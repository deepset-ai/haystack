# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from haystack.utils.dataclasses import _warn_on_inplace_mutation


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
    snapshot_file_path: str | None = None

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


@_warn_on_inplace_mutation
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


@_warn_on_inplace_mutation
@dataclass
class PipelineSnapshot:
    """
    A dataclass to hold a snapshot of the pipeline at a specific point in time.

    :param original_input_data: The original input data provided to the pipeline.
    :param ordered_component_names: A list of component names in the order they were visited.
    :param pipeline_state: The state of the pipeline at the time of the snapshot.
    :param break_point: The breakpoint that triggered the snapshot.
    :param timestamp: A timestamp indicating when the snapshot was taken.
    :param include_outputs_from: Set of component names whose outputs should be included in the pipeline results.
    """

    original_input_data: dict[str, Any]
    ordered_component_names: list[str]
    pipeline_state: PipelineState
    break_point: Breakpoint
    timestamp: datetime | None = None
    include_outputs_from: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
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
        return {
            "pipeline_state": self.pipeline_state.to_dict(),
            "break_point": self.break_point.to_dict(),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "original_input_data": self.original_input_data,
            "ordered_component_names": self.ordered_component_names,
            "include_outputs_from": list(self.include_outputs_from),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineSnapshot":
        """
        Populate the PipelineSnapshot from a dictionary representation.

        :param data: A dictionary containing the pipeline state, timestamp, breakpoint, agent snapshot, original input
                     data, ordered component names, include_outputs_from, and pipeline outputs.
        """
        include_outputs_from = set(data.get("include_outputs_from", []))

        return cls(
            pipeline_state=PipelineState.from_dict(data=data["pipeline_state"]),
            break_point=Breakpoint.from_dict(data=data["break_point"]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            original_input_data=data.get("original_input_data", {}),
            ordered_component_names=data.get("ordered_component_names", []),
            include_outputs_from=include_outputs_from,
        )
