# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin

from docstring_parser import parse
from pydantic import TypeAdapter

from haystack import logging
from haystack.core.component import Component
from haystack.tools import Tool

logger = logging.getLogger(__name__)


def is_pydantic_v2_model(instance: Any) -> bool:
    """
    Checks if the instance is a Pydantic v2 model.

    :param instance: The instance to check.
    :returns: True if the instance is a Pydantic v2 model, False otherwise.
    """
    return hasattr(instance, "model_validate")


class ComponentTool(Tool):
    """
    A Tool that wraps Haystack components, allowing them to be used as tools by LLMs.

    ComponentTool automatically generates LLM-compatible tool schemas from Component input sockets,
    which are derived from the component's `run` method signature and type hints.


    Key features:
    - Automatic LLM tool calling schema generation from Component input sockets
    - Type conversion and validation for Component inputs
    - Support for complex types (dataclasses, Pydantic models, lists)
    - Automatic name generation from Component class name
    - Description extraction from Component docstrings

    Let's assume we already have (or want to create) a Haystack component that we want to use as a tool.
    We can create a ComponentTool from the component by passing the component to the ComponentTool constructor.

    ```python
    from haystack import component
    from haystack.tools import ComponentTool

    @component
    class WeatherComponent:
        '''Gets weather information for a location.'''

        def run(self, city: str, units: str = "celsius"):
            '''
            :param city: The city to get weather for
            :param units: Temperature units (celsius/fahrenheit)
            '''
            return f"Weather in {city}: 20°{units}"

    # Create a tool from the component
    weather = WeatherComponent()
    tool = ComponentTool(
        component=weather,
        name="get_weather",  # Optional: defaults to snake_case of class name
        description="Get current weather for a city"  # Optional: defaults to component run method docstring
    )
    ```

    """

    def __init__(self, component: Component, name: Optional[str] = None, description: Optional[str] = None):
        """
        Create a Tool instance from a Haystack component.

        :param component: The Haystack Component to wrap as a tool
        :param name: Optional name for the tool (defaults to snake_case of Component class name)
        :param description: Optional description (defaults to Component's docstring)
        :raises ValueError: If the component is invalid or schema generation fails
        """
        if not isinstance(component, Component):
            message = (
                f"Object {component!r} is not a Haystack component. "
                "Use this method to create a Tool only with Haystack component instances."
            )
            raise ValueError(message)

        # Create the tools schema from the component run method parameters
        tool_schema = self._create_tool_parameters_schema(component)

        def component_invoker(**kwargs):
            """
            Invokes the component using keyword arguments provided by the LLM function calling/tool generated response.

            :param kwargs: The keyword arguments to invoke the component with.
            :returns: The result of the component invocation.
            """
            converted_kwargs = {}
            input_sockets = component.__haystack_input__._sockets_dict
            for param_name, param_value in kwargs.items():
                param_type = input_sockets[param_name].type

                # Check if the type (or list element type) has from_dict
                target_type = get_args(param_type)[0] if get_origin(param_type) is list else param_type
                if hasattr(target_type, "from_dict"):
                    if isinstance(param_value, list):
                        param_value = [target_type.from_dict(item) for item in param_value if isinstance(item, dict)]
                    elif isinstance(param_value, dict):
                        param_value = target_type.from_dict(param_value)
                else:
                    # Let TypeAdapter handle both single values and lists
                    type_adapter = TypeAdapter(param_type)
                    param_value = type_adapter.validate_python(param_value)

                converted_kwargs[param_name] = param_value
            logger.debug(f"Invoking component {type(component)} with kwargs: {converted_kwargs}")
            return component.run(**converted_kwargs)

        # Generate a name for the tool if not provided
        if not name:
            class_name = component.__class__.__name__
            # Convert camelCase/PascalCase to snake_case
            name = "".join(
                [
                    "_" + c.lower() if c.isupper() and i > 0 and not class_name[i - 1].isupper() else c.lower()
                    for i, c in enumerate(class_name)
                ]
            ).lstrip("_")

        # Generate a description for the tool if not provided and truncate to 512 characters
        # as most LLMs have a limit for the description length
        description = (description or component.__doc__ or name)[:512]

        # Create the Tool instance with the component invoker as the function to be called and the schema
        super().__init__(name, description, tool_schema, component_invoker)
        self.component = component

    def _create_tool_parameters_schema(self, component: Component) -> Dict[str, Any]:
        """
        Creates an OpenAI tools schema from a component's run method parameters.

        :param component: The component to create the schema from.
        :returns: OpenAI tools schema for the component's run method parameters.
        """
        properties = {}
        required = []

        param_descriptions = self._get_param_descriptions(component.run)

        for input_name, socket in component.__haystack_input__._sockets_dict.items():  # type: ignore[attr-defined]
            input_type = socket.type
            description = param_descriptions.get(input_name, f"Input '{input_name}' for the component.")

            try:
                property_schema = self._create_property_schema(input_type, description)
            except ValueError as e:
                raise ValueError(f"Error processing input '{input_name}': {e}")

            properties[input_name] = property_schema

            # Use socket.is_mandatory to check if the input is required
            if socket.is_mandatory:
                required.append(input_name)

        parameters_schema = {"type": "object", "properties": properties}

        if required:
            parameters_schema["required"] = required

        return parameters_schema

    def _get_param_descriptions(self, method: Callable) -> Dict[str, str]:
        """
        Extracts parameter descriptions from the method's docstring using docstring_parser.

        :param method: The method to extract parameter descriptions from.
        :returns: A dictionary mapping parameter names to their descriptions.
        """
        docstring = getdoc(method)
        if not docstring:
            return {}

        parsed_doc = parse(docstring)
        param_descriptions = {}
        for param in parsed_doc.params:
            if not param.description:
                logger.warning(
                    "Missing description for parameter '%s'. Please add a description in the component's "
                    "run() method docstring using the format ':param %%s: <description>'. "
                    "This description helps the LLM understand how to use this parameter." % param.arg_name
                )
            param_descriptions[param.arg_name] = param.description.strip() if param.description else ""
        return param_descriptions

    def _is_nullable_type(self, python_type: Any) -> bool:
        """
        Checks if the type is a Union with NoneType (i.e., Optional).

        :param python_type: The Python type to check.
        :returns: True if the type is a Union with NoneType, False otherwise.
        """
        origin = get_origin(python_type)
        if origin is Union:
            return type(None) in get_args(python_type)
        return False

    def _create_list_schema(self, item_type: Any, description: str) -> Dict[str, Any]:
        """
        Creates a schema for a list type.

        :param item_type: The type of items in the list.
        :param description: The description of the list.
        :returns: A dictionary representing the list schema.
        """
        items_schema = self._create_property_schema(item_type, "")
        items_schema.pop("description", None)
        return {"type": "array", "description": description, "items": items_schema}

    def _create_dataclass_schema(self, python_type: Any, description: str) -> Dict[str, Any]:
        """
        Creates a schema for a dataclass.

        :param python_type: The dataclass type.
        :param description: The description of the dataclass.
        :returns: A dictionary representing the dataclass schema.
        """
        schema = {"type": "object", "description": description, "properties": {}}
        cls = python_type if isinstance(python_type, type) else python_type.__class__
        for field in fields(cls):
            field_description = f"Field '{field.name}' of '{cls.__name__}'."
            if isinstance(schema["properties"], dict):
                schema["properties"][field.name] = self._create_property_schema(field.type, field_description)
        return schema

    def _create_pydantic_schema(self, python_type: Any, description: str) -> Dict[str, Any]:
        """
        Creates a schema for a Pydantic model.

        :param python_type: The Pydantic model type.
        :param description: The description of the model.
        :returns: A dictionary representing the Pydantic model schema.
        """
        schema = {"type": "object", "description": description, "properties": {}}
        required_fields = []

        for m_name, m_field in python_type.model_fields.items():
            field_description = f"Field '{m_name}' of '{python_type.__name__}'."
            if isinstance(schema["properties"], dict):
                schema["properties"][m_name] = self._create_property_schema(m_field.annotation, field_description)
            if m_field.is_required():
                required_fields.append(m_name)

        if required_fields:
            schema["required"] = required_fields
        return schema

    def _create_basic_type_schema(self, python_type: Any, description: str) -> Dict[str, Any]:
        """
        Creates a schema for a basic Python type.

        :param python_type: The Python type.
        :param description: The description of the type.
        :returns: A dictionary representing the basic type schema.
        """
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}
        return {"type": type_mapping.get(python_type, "string"), "description": description}

    def _create_property_schema(self, python_type: Any, description: str, default: Any = None) -> Dict[str, Any]:
        """
        Creates a property schema for a given Python type, recursively if necessary.

        :param python_type: The Python type to create a property schema for.
        :param description: The description of the property.
        :param default: The default value of the property.
        :returns: A dictionary representing the property schema.
        """
        nullable = self._is_nullable_type(python_type)
        if nullable:
            non_none_types = [t for t in get_args(python_type) if t is not type(None)]
            python_type = non_none_types[0] if non_none_types else str

        origin = get_origin(python_type)
        if origin is list:
            schema = self._create_list_schema(get_args(python_type)[0] if get_args(python_type) else Any, description)
        elif is_dataclass(python_type):
            schema = self._create_dataclass_schema(python_type, description)
        elif is_pydantic_v2_model(python_type):
            schema = self._create_pydantic_schema(python_type, description)
        else:
            schema = self._create_basic_type_schema(python_type, description)

        if default is not None:
            schema["default"] = default

        return schema
