# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Type


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    def __init__(self, component_name: str, component_type: Type, detail: str) -> None:
        self.component_name = component_name
        self.component_type = component_type
        super().__init__(detail)

    def __str__(self):
        return (
            f"PipelineRuntimeError:\n"
            f"Component name: '{self.component_name}'\n"
            f"Component type: '{self.component_type.__name__}'\n"
            f"Details: {self.args[0]}"
        )

    @classmethod
    def from_exception(cls, component_name: str, component_type: Type, error: Exception):
        """
        Create a PipelineRuntimeError from an exception.
        """
        return cls(component_name, component_type, f"Failed to run component. Error: {str(error)}")

    @classmethod
    def from_invalid_output(cls, component_name: str, component_type: Type, output: Any):
        """
        Create a PipelineRuntimeError from an invalid output.
        """
        return cls(component_name, component_type, f"Invalid output type: {type(output)}. Expected a dictionary.")


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
