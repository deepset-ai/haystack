# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.errors import ToolInvocationError
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


@dataclass
class Tool:
    """
    Data class representing a Tool that Language Models can prepare a call for.

    Accurate definitions of the textual attributes such as `name` and `description`
    are important for the Language Model to correctly prepare the call.

    For resource-intensive operations like establishing connections to remote services or
    loading models, override the `warm_up()` method. This method is called before the Tool
    is used and should be idempotent, as it may be called multiple times during
    pipeline/agent setup.

    :param name:
        Name of the Tool.
    :param description:
        Description of the Tool.
    :param parameters:
        A JSON schema defining the parameters expected by the Tool.
    :param function:
        The function that will be invoked when the Tool is called.
    :param outputs_to_string:
        Optional dictionary defining how a tool outputs should be converted into a string.
        If the source is provided only the specified output key is sent to the handler.
        If the source is omitted the whole tool result is sent to the handler.
        Example:
        ```python
        {
            "source": "docs", "handler": format_documents
        }
        ```
    :param inputs_from_state:
        Optional dictionary mapping state keys to tool parameter names.
        Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
    :param outputs_to_state:
        Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
        If the source is provided only the specified output key is sent to the handler.
        Example:
        ```python
        {
            "documents": {"source": "docs", "handler": custom_handler}
        }
        ```
        If the source is omitted the whole tool result is sent to the handler.
        Example:
        ```python
        {
            "documents": {"handler": custom_handler}
        }
        ```
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable
    outputs_to_string: Optional[dict[str, Any]] = None
    inputs_from_state: Optional[dict[str, str]] = None
    outputs_to_state: Optional[dict[str, dict[str, Any]]] = None

    def __post_init__(self):
        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

        # Validate outputs structure if provided
        if self.outputs_to_state is not None:
            for key, config in self.outputs_to_state.items():
                if not isinstance(config, dict):
                    raise ValueError(f"outputs_to_state configuration for key '{key}' must be a dictionary")
                if "source" in config and not isinstance(config["source"], str):
                    raise ValueError(f"outputs_to_state source for key '{key}' must be a string.")
                if "handler" in config and not callable(config["handler"]):
                    raise ValueError(f"outputs_to_state handler for key '{key}' must be callable")

        if self.outputs_to_string is not None:
            if "source" in self.outputs_to_string and not isinstance(self.outputs_to_string["source"], str):
                raise ValueError("outputs_to_string source must be a string.")
            if "handler" in self.outputs_to_string and not callable(self.outputs_to_string["handler"]):
                raise ValueError("outputs_to_string handler must be callable")

    @property
    def tool_spec(self) -> dict[str, Any]:
        """
        Return the Tool specification to be used by the Language Model.
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def warm_up(self) -> None:
        """
        Prepare the Tool for use.

        Override this method to establish connections to remote services, load models,
        or perform other resource-intensive initialization. This method should be idempotent,
        as it may be called multiple times.
        """
        pass

    def invoke(self, **kwargs: Any) -> Any:
        """
        Invoke the Tool with the provided keyword arguments.
        """
        try:
            result = self.function(**kwargs)
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke Tool `{self.name}` with parameters {kwargs}. Error: {e}", tool_name=self.name
            ) from e
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = asdict(self)
        data["function"] = serialize_callable(self.function)

        if self.outputs_to_state is not None:
            data["outputs_to_state"] = _serialize_outputs_to_state(self.outputs_to_state)

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            # This is soft-copied as to not modify the attributes in place
            data["outputs_to_string"] = self.outputs_to_string.copy()
            data["outputs_to_string"]["handler"] = serialize_callable(self.outputs_to_string["handler"])
        else:
            data["outputs_to_string"] = None

        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        init_parameters = data["data"]
        init_parameters["function"] = deserialize_callable(init_parameters["function"])
        if "outputs_to_state" in init_parameters and init_parameters["outputs_to_state"]:
            init_parameters["outputs_to_state"] = _deserialize_outputs_to_state(init_parameters["outputs_to_state"])

        if (
            init_parameters.get("outputs_to_string") is not None
            and init_parameters["outputs_to_string"].get("handler") is not None
        ):
            init_parameters["outputs_to_string"]["handler"] = deserialize_callable(
                init_parameters["outputs_to_string"]["handler"]
            )

        return cls(**init_parameters)


def _check_duplicate_tool_names(tools: Optional[list[Tool]]) -> None:
    """
    Checks for duplicate tool names and raises a ValueError if they are found.

    :param tools: The list of tools to check.
    :raises ValueError: If duplicate tool names are found.
    """
    if tools is None:
        return
    tool_names = [tool.name for tool in tools]
    duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
    if duplicate_tool_names:
        raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")


def _serialize_outputs_to_state(outputs_to_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Serializes the outputs_to_state dictionary, converting any callable handlers to their string representation.

    :param outputs_to_state: The outputs_to_state dictionary to serialize.
    :returns: The serialized outputs_to_state dictionary.
    """
    serialized_outputs = {}
    for key, config in outputs_to_state.items():
        serialized_config = config.copy()
        if "handler" in config:
            serialized_config["handler"] = serialize_callable(config["handler"])
        serialized_outputs[key] = serialized_config
    return serialized_outputs


def _deserialize_outputs_to_state(outputs_to_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Deserializes the outputs_to_state dictionary, converting any string handlers back to callables.

    :param outputs_to_state: The outputs_to_state dictionary to deserialize.
    :returns: The deserialized outputs_to_state dictionary.
    """
    deserialized_outputs = {}
    for key, config in outputs_to_state.items():
        deserialized_config = config.copy()
        if "handler" in config:
            deserialized_config["handler"] = deserialize_callable(config["handler"])
        deserialized_outputs[key] = deserialized_config
    return deserialized_outputs
