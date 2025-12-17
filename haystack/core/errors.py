# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, PipelineSnapshot, ToolBreakpoint


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    def __init__(
        self,
        component_name: str | None,
        component_type: type | None,
        message: str,
        pipeline_snapshot: PipelineSnapshot | None = None,
        *,
        pipeline_snapshot_file_path: str | None = None,
    ) -> None:
        self.component_name = component_name
        self.component_type = component_type
        self.pipeline_snapshot = pipeline_snapshot
        self.pipeline_snapshot_file_path = pipeline_snapshot_file_path
        super().__init__(message)

    @classmethod
    def from_exception(cls, component_name: str, component_type: type, error: Exception) -> "PipelineRuntimeError":
        """
        Create a PipelineRuntimeError from an exception.
        """
        message = (
            f"The following component failed to run:\n"
            f"Component name: '{component_name}'\n"
            f"Component type: '{component_type.__name__}'\n"
            f"Error: {str(error)}"
        )
        return cls(component_name, component_type, message)

    @classmethod
    def from_invalid_output(cls, component_name: str, component_type: type, output: Any) -> "PipelineRuntimeError":
        """
        Create a PipelineRuntimeError from an invalid output.
        """
        message = (
            f"The following component returned an invalid output:\n"
            f"Component name: '{component_name}'\n"
            f"Component type: '{component_type.__name__}'\n"
            f"Expected a dictionary, but got {type(output).__name__} instead.\n"
            f"Check the component's output and ensure it is a valid dictionary."
        )
        return cls(component_name, component_type, message)


class PipelineComponentsBlockedError(PipelineRuntimeError):
    def __init__(self) -> None:
        message = (
            "Cannot run pipeline - all components are blocked. "
            "This typically happens when:\n"
            "1. There is no valid entry point for the pipeline\n"
            "2. There is a circular dependency preventing the pipeline from running\n"
            "Check the connections between these components and ensure all required inputs are provided."
        )
        super().__init__(None, None, message)


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineDrawingError(PipelineError):
    pass


class PipelineMaxComponentRuns(PipelineError):
    pass


class PipelineUnmarshalError(PipelineError):
    pass


class ComponentError(Exception):
    pass


class ComponentDeserializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


class SerializationError(Exception):
    pass


class BreakpointException(Exception):
    """
    Exception raised when a pipeline breakpoint is triggered.
    """

    def __init__(
        self,
        message: str,
        component: str | None = None,
        pipeline_snapshot: PipelineSnapshot | None = None,
        pipeline_snapshot_file_path: str | None = None,
        *,
        break_point: AgentBreakpoint | Breakpoint | ToolBreakpoint | None = None,
    ):
        super().__init__(message)
        self.component = component
        self.pipeline_snapshot = pipeline_snapshot
        self.pipeline_snapshot_file_path = pipeline_snapshot_file_path
        self._break_point = break_point

        if self.pipeline_snapshot is None and self._break_point is None:
            raise ValueError("Either pipeline_snapshot or break_point must be provided.")

    @classmethod
    def from_triggered_breakpoint(cls, break_point: Breakpoint | ToolBreakpoint) -> "BreakpointException":
        """
        Create a BreakpointException from a triggered breakpoint.
        """
        msg = f"Breaking at component {break_point.component_name} at visit count {break_point.visit_count}"
        return BreakpointException(message=msg, component=break_point.component_name, break_point=break_point)

    @property
    def inputs(self):
        """
        Returns the inputs of the pipeline or agent at the breakpoint.

        If an AgentBreakpoint caused this exception, returns the inputs of the agent's internal components.
        Otherwise, returns the current inputs of the pipeline.
        """
        if not self.pipeline_snapshot:
            return None

        if self.pipeline_snapshot.agent_snapshot:
            return self.pipeline_snapshot.agent_snapshot.component_inputs
        return self.pipeline_snapshot.pipeline_state.inputs

    @property
    def results(self):
        """
        Returns the results of the pipeline or agent at the breakpoint.

        If an AgentBreakpoint caused this exception, returns the current results of the agent.
        Otherwise, returns the current outputs of the pipeline.
        """
        if not self.pipeline_snapshot:
            return None

        if self.pipeline_snapshot.agent_snapshot:
            return self.pipeline_snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["state"]
        return self.pipeline_snapshot.pipeline_state.pipeline_outputs

    @property
    def break_point(self) -> AgentBreakpoint | Breakpoint | ToolBreakpoint:
        """
        Returns the Breakpoint or AgentBreakpoint that caused this exception, if available.

        If a specific break point was provided during initialization, it is returned.
        Otherwise, if the pipeline snapshot contains a break point, that is returned.
        """
        if self._break_point is not None:
            return self._break_point
        # Mypy doesn't know that pipeline_snapshot.break_point must not be None here based on the constructor check
        return self.pipeline_snapshot.break_point  # type: ignore[union-attr]


class PipelineInvalidPipelineSnapshotError(Exception):
    """
    Exception raised when a pipeline is resumed from an invalid snapshot.
    """

    def __init__(self, message: str):
        super().__init__(message)
