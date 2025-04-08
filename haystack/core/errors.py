# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Type


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    def __init__(self, component_name: Optional[str], component_type: Optional[Type], message: str) -> None:
        self.component_name = component_name
        self.component_type = component_type
        super().__init__(message)

    @classmethod
    def from_exception(cls, component_name: str, component_type: Type, error: Exception):
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
    def from_invalid_output(cls, component_name: str, component_type: Type, output: Any):
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
