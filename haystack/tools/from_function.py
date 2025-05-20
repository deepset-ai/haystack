# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Callable, Dict, Optional, Union

from pydantic import create_model

from .errors import SchemaGenerationError
from .tool import Tool


def create_tool_from_function(
    function: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    inputs_from_state: Optional[Dict[str, str]] = None,
    outputs_to_state: Optional[Dict[str, Dict[str, Any]]] = None,
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
    :param inputs_from_state:
        Optional dictionary mapping state keys to tool parameter names.
        Example: {"repository": "repo"} maps state's "repository" to tool's "repo" parameter.
    :param outputs_to_state:
        Optional dictionary defining how tool outputs map to state and message handling.
        Example: {
            "documents": {"source": "docs", "handler": custom_handler},
            "message": {"source": "summary", "handler": format_summary}
        }
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
        # Skip adding parameter names that will be passed to the tool from State
        if inputs_from_state and param_name in inputs_from_state.values():
            continue

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

    return Tool(
        name=name or function.__name__,
        description=tool_description,
        parameters=schema,
        function=function,
        inputs_from_state=inputs_from_state,
        outputs_to_state=outputs_to_state,
    )


def tool(
    function: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    inputs_from_state: Optional[Dict[str, str]] = None,
    outputs_to_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Union[Tool, Callable[[Callable], Tool]]:
    """
    Decorator to convert a function into a Tool.

    Can be used with or without parameters:
    @tool  # without parameters
    def my_function(): ...

    @tool(name="custom_name")  # with parameters
    def my_function(): ...

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

    :param function: The function to decorate (when used without parameters)
    :param name: Optional custom name for the tool
    :param description: Optional custom description
    :param inputs_from_state: Optional dictionary mapping state keys to tool parameter names
    :param outputs_to_state: Optional dictionary defining how tool outputs map to state and message handling
    :return: Either a Tool instance or a decorator function that will create one
    """

    def decorator(func: Callable) -> Tool:
        return create_tool_from_function(
            function=func,
            name=name,
            description=description,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
        )

    if function is None:
        return decorator
    return decorator(function)


def _remove_title_from_schema(schema: Dict[str, Any]):
    """
    Remove the 'title' keyword from JSON schema and contained property schemas.

    :param schema:
        The JSON schema to remove the 'title' keyword from.
    """
    for key, value in list(schema.items()):
        # Make sure not to remove parameters named title
        if key == "properties" and isinstance(value, dict) and "title" in value:
            for sub_val in value.values():
                _remove_title_from_schema(sub_val)
        elif key == "title":
            del schema[key]
        elif isinstance(value, dict):
            _remove_title_from_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _remove_title_from_schema(item)
