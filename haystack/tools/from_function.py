# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import textwrap
from typing import Any, Callable, Dict, Optional

from pydantic import create_model

from haystack.tools.errors import SchemaGenerationError
from haystack.tools.tool import Tool

# Define constants for ReST directives
REST_DIRECTIVES = r"param|return|returns|raise|raises"


def create_tool_from_function(
    function: Callable, name: Optional[str] = None, description: Optional[str] = None
) -> "Tool":
    """
    Create a Tool instance from a function.

    Allows customizing the Tool name and description.
    For simpler use cases, consider using the `@tool` decorator.

    ### Usage example

    ```python
    from typing import Annotated, Literal
    from haystack.tools import create_tool_from_function

    def get_weather(
        city: Annotated[str, "the city for which to get the weather"] = "Munich",
        unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
        '''A simple function to get the current weather for a location.'''
        return f"Weather report for {city}: 20 {unit}, sunny"

    tool = create_tool_from_function(get_weather)

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
        The function is expected to have basic python input types (str, int, float, bool, list, dict, tuple).
        Other input types may work but are not guaranteed.
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

    tool_description = ""
    param_descriptions_from_rest: Dict[str, str] = {}
    return_description = ""
    raises_descriptions = []

    if description is not None:
        tool_description = description
    else:
        # Process docstring if available
        if function.__doc__:
            docstring = textwrap.dedent(function.__doc__).strip()

            # Check if this is a ReST-style docstring
            if re.search(rf":({REST_DIRECTIVES})\s+", docstring):
                # Extract main description (everything before first directive)
                main_parts = re.split(rf":({REST_DIRECTIVES})\s+", docstring, 1)
                tool_description = main_parts[0].strip()

                # Parse parameter descriptions (handling both :param name: and :param type name: formats)
                param_pattern = re.compile(rf":param\s+(\w+)\s*:(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL)
                param_descriptions_from_rest = {name: desc.strip() for name, desc in param_pattern.findall(docstring)}

                # Parse return descriptions
                return_pattern = re.compile(rf":return:\s*(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL)
                return_matches = return_pattern.findall(docstring)
                if return_matches:
                    return_description = return_matches[0].strip()

                # Parse raises descriptions
                raises_pattern = re.compile(
                    rf":raises?\s+(\w+(?:,\s*\w+)*)\s*:\s*(.*?)(?=:(?:{REST_DIRECTIVES})|$)", re.DOTALL
                )
                for exc_types, desc in raises_pattern.findall(docstring):
                    for exc_type in re.split(r",\s*", exc_types):
                        raises_descriptions.append(f"{exc_type}: {desc.strip()}")
            else:
                # Not a ReST-style docstring, use the whole thing
                tool_description = docstring.strip()

    # Build a comprehensive description including return values and exceptions
    full_description = tool_description

    if return_description:
        full_description += f"\n\nReturns: {return_description}"

    if raises_descriptions:
        full_description += "\n\nRaises:\n" + "\n".join(f"- {r}" for r in raises_descriptions)

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

        # Priority 1: Get descriptions from Annotated type hints
        if hasattr(param.annotation, "__metadata__"):
            descriptions[param_name] = param.annotation.__metadata__[0]
        # Priority 2: Get descriptions from ReST docstring
        elif param_name in param_descriptions_from_rest:
            descriptions[param_name] = param_descriptions_from_rest[param_name]

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

    return Tool(name=name or function.__name__, description=full_description, parameters=schema, function=function)


def tool(function: Callable) -> Tool:
    """
    Decorator to convert a function into a Tool.

    Tool name, description, and parameters are inferred from the function.
    If you need to customize more the Tool, use `create_tool_from_function` instead.

    ### Usage example
    ```python
    from typing import Annotated, Literal
    from haystack.tools import tool

    @tool
    def get_weather(
        city: Annotated[str, "the city for which to get the weather"] = "Munich",
        unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
        '''A simple function to get the current weather for a location.'''
        return f"Weather report for {city}: 20 {unit}, sunny"

    print(get_weather)
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
    """
    return create_tool_from_function(function)


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
