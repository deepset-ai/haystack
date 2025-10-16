# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Optional

from haystack.dataclasses.breakpoints import PipelineSnapshot


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    def __init__(
        self,
        component_name: Optional[str],
        component_type: Optional[type],
        message: str,
        pipeline_snapshot: Optional[PipelineSnapshot] = None,
        pipeline_snapshot_file_path: Optional[str] = None,
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
        component: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        results: Optional[dict[str, Any]] = None,
        pipeline_snapshot: Optional[PipelineSnapshot] = None,
        pipeline_snapshot_file_path: Optional[str] = None,
    ):
        super().__init__(message)
        self.component = component
        self.pipeline_snapshot = pipeline_snapshot
        self.pipeline_snapshot_file_path = pipeline_snapshot_file_path

        self.inputs = inputs
        self.results = results
        warnings.warn(
            "The `inputs` and `results` parameters will be removed in the 2.20.0 release. "
            "Please use the `pipeline_snapshot` to access this information.",
            DeprecationWarning,
        )


class PipelineInvalidPipelineSnapshotError(Exception):
    """
    Exception raised when a pipeline is resumed from an invalid snapshot.
    """

    def __init__(self, message: str):
        super().__init__(message)
