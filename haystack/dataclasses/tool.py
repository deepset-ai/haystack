# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import create_model

from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, serialize_callable

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
    """

    pass


class SchemaGenerationError(Exception):
    """
    Exception raised when automatic schema generation fails.
    """

    pass


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

        serialized = asdict(self)
        serialized["function"] = serialize_callable(self.function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        data["function"] = deserialize_callable(data["function"])
        return cls(**data)

    @classmethod
    def from_function(cls, function: Callable, name: Optional[str] = None, description: Optional[str] = None) -> "Tool":
        """
        Create a Tool instance from a function.

        ### Usage example

        ```python
        from typing import Annotated, Literal
        from haystack.dataclasses import Tool

        def get_weather(
            city: Annotated[str, "the city for which to get the weather"] = "Munich",
            unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
            '''A simple function to get the current weather for a location.'''
            return f"Weather report for {city}: 20 {unit}, sunny"

        tool = Tool.from_function(get_weather)

        print(tool)
        >>> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
        >>> parameters={
        >>> 'type': 'object',
        >>> 'properties': {
        >>>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
        >>>     'unit': {
        >>>         'type': 'string',
        >>>         'enum': ['Celsius', 'Fahrenheit'],
        >>>         'description': 'the unit for the temperature',
        >>>         'default': 'Celsius',
        >>>     },
        >>>     }
        >>> },
        >>> function=<function get_weather at 0x7f7b3a8a9b80>)
        ```

        :param function:
            The function to be converted into a Tool.
            The function must include type hints for all parameters.
            If a parameter is annotated using `typing.Annotated`, its metadata will be used as parameter description.
        :param name:
            The name of the Tool. If not provided, the name of the function will be used.
        :param description:
            The description of the Tool. If not provided, the docstring of the function will be used.
            To intentionally leave the description empty, pass an empty string.

        :returns:
            The Tool created from the function.

        :raises ValueError:
            If any parameter of the function lacks a type hint.
        :raises SchemaGenerationError:
            If there is an error generating the JSON schema for the Tool.
        """

        tool_description = description if description is not None else (function.__doc__ or "")

        signature = inspect.signature(function)

        # collect fields (types and defaults) and descriptions from function parameters
        fields: Dict[str, Any] = {}
        descriptions = {}

        for param_name, param in signature.parameters.items():
            if param.annotation is param.empty:
                raise ValueError(f"Function '{function.__name__}': parameter '{param_name}' does not have a type hint.")

            # if the parameter has not a default value, Pydantic requires an Ellipsis (...)
            # to explicitly indicate that the parameter is required
            default = param.default if param.default is not param.empty else ...
            fields[param_name] = (param.annotation, default)

            if hasattr(param.annotation, "__metadata__"):
                descriptions[param_name] = param.annotation.__metadata__[0]

        # create Pydantic model and generate JSON schema
        try:
            model = create_model(function.__name__, **fields)
            schema = model.model_json_schema()
        except Exception as e:
            raise SchemaGenerationError(f"Failed to create JSON schema for function '{function.__name__}'") from e

        # we don't want to include title keywords in the schema, as they contain redundant information
        # there is no programmatic way to prevent Pydantic from adding them, so we remove them later
        # see https://github.com/pydantic/pydantic/discussions/8504
        _remove_title_from_schema(schema)

        # add parameters descriptions to the schema
        for param_name, param_description in descriptions.items():
            if param_name in schema["properties"]:
                schema["properties"][param_name]["description"] = param_description

        return Tool(name=name or function.__name__, description=tool_description, parameters=schema, function=function)


def _remove_title_from_schema(schema: Dict[str, Any]):
    """
    Remove the 'title' keyword from JSON schema and contained property schemas.

    :param schema:
        The JSON schema to remove the 'title' keyword from.
    """
    schema.pop("title", None)

    for property_schema in schema["properties"].values():
        for key in list(property_schema.keys()):
            if key == "title":
                del property_schema[key]


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
            deserialized_tools.append(Tool.from_dict(tool))

        data[key] = deserialized_tools
