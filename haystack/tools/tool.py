# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.lazy_imports import LazyImport
from haystack.tools.errors import ToolInvocationError
from haystack.utils import deserialize_callable, serialize_callable

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError


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
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def __post_init__(self):
        jsonschema_import.check()
        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

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
            raise ToolInvocationError(f"Failed to invoke Tool `{self.name}` with parameters {kwargs}") from e
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        data = asdict(self)
        data["function"] = serialize_callable(self.function)
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


def deserialize_tools_inplace(data: Dict[str, Any], key: str = "tools"):
    """
    Deserialize Tools in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the Tools are stored.
    """
    if key in data:
        serialized_tools = data[key]

        if serialized_tools is None:
            return

        if not isinstance(serialized_tools, list):
            raise TypeError(f"The value of '{key}' is not a list")

        deserialized_tools = []
        for tool in serialized_tools:
            if not isinstance(tool, dict):
                raise TypeError(f"Serialized tool '{tool}' is not a dictionary")

            # different classes are allowed: Tool, ComponentTool, etc.
            tool_class = import_class_by_name(tool["type"])
            if not issubclass(tool_class, Tool):
                raise TypeError(f"Class '{tool_class}' is not a subclass of Tool")

            deserialized_tools.append(tool_class.from_dict(tool))

        data[key] = deserialized_tools
