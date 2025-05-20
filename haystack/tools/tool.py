# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.errors import ToolInvocationError
from haystack.utils import deserialize_callable, serialize_callable


@dataclass
class Tool:
    """
    Data class representing a Tool that Language Models can prepare a call for.

    Accurate definitions of the textual attributes such as `name` and `description`
    are important for the Language Model to correctly prepare the call.

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
        Example: {
            "source": "docs", "handler": format_documents
        }
    :param inputs_from_state:
        Optional dictionary mapping state keys to tool parameter names.
        Example: {"repository": "repo"} maps state's "repository" to tool's "repo" parameter.
    :param outputs_to_state:
        Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
        If the source is provided only the specified output key is sent to the handler.
        Example: {
            "documents": {"source": "docs", "handler": custom_handler}
        }
        If the source is omitted the whole tool result is sent to the handler.
        Example: {
            "documents": {"handler": custom_handler}
        }
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    outputs_to_string: Optional[Dict[str, Any]] = None
    inputs_from_state: Optional[Dict[str, str]] = None
    outputs_to_state: Optional[Dict[str, Dict[str, Any]]] = None

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
    def tool_spec(self) -> Dict[str, Any]:
        """
        Return the Tool specification to be used by the Language Model.
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs) -> Any:
        """
        Invoke the Tool with the provided keyword arguments.
        """
        try:
            result = self.function(**kwargs)
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke Tool `{self.name}` with parameters {kwargs}. Error: {e}"
            ) from e
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = asdict(self)
        data["function"] = serialize_callable(self.function)

        # Serialize output handlers if they exist
        if self.outputs_to_state:
            serialized_outputs = {}
            for key, config in self.outputs_to_state.items():
                serialized_config = config.copy()
                if "handler" in config:
                    serialized_config["handler"] = serialize_callable(config["handler"])
                serialized_outputs[key] = serialized_config
            data["outputs_to_state"] = serialized_outputs

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            data["outputs_to_string"] = serialize_callable(self.outputs_to_string["handler"])

        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        init_parameters = data["data"]
        init_parameters["function"] = deserialize_callable(init_parameters["function"])

        # Deserialize output handlers if they exist
        if "outputs_to_state" in init_parameters and init_parameters["outputs_to_state"]:
            deserialized_outputs = {}
            for key, config in init_parameters["outputs_to_state"].items():
                deserialized_config = config.copy()
                if "handler" in config:
                    deserialized_config["handler"] = deserialize_callable(config["handler"])
                deserialized_outputs[key] = deserialized_config
            init_parameters["outputs_to_state"] = deserialized_outputs

        if (
            init_parameters.get("outputs_to_string") is not None
            and init_parameters["outputs_to_string"].get("handler") is not None
        ):
            init_parameters["outputs_to_string"]["handler"] = deserialize_callable(
                init_parameters["outputs_to_string"]["handler"]
            )

        return cls(**init_parameters)


def _check_duplicate_tool_names(tools: Optional[List[Tool]]) -> None:
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
