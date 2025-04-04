# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy, deepcopy
from dataclasses import fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin

from pydantic import TypeAdapter

from haystack import logging
from haystack.core.component import Component
from haystack.core.serialization import (
    component_from_dict,
    component_to_dict,
    generate_qualified_class_name,
    import_class_by_name,
)
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool
from haystack.tools.errors import SchemaGenerationError
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

with LazyImport(message="Run 'pip install docstring-parser'") as docstring_parser_import:
    from docstring_parser import parse


logger = logging.getLogger(__name__)


class ComponentTool(Tool):
    """
    A Tool that wraps Haystack components, allowing them to be used as tools by LLMs.

    ComponentTool automatically generates LLM-compatible tool schemas from component input sockets,
    which are derived from the component's `run` method signature and type hints.


    Key features:
    - Automatic LLM tool calling schema generation from component input sockets
    - Type conversion and validation for component inputs
    - Support for types:
        - Dataclasses
        - Lists of dataclasses
        - Basic types (str, int, float, bool, dict)
        - Lists of basic types
    - Automatic name generation from component class name
    - Description extraction from component docstrings

    To use ComponentTool, you first need a Haystack component - either an existing one or a new one you create.
    You can create a ComponentTool from the component by passing the component to the ComponentTool constructor.
    Below is an example of creating a ComponentTool from an existing SerperDevWebSearch component.

    ```python
    from haystack import component, Pipeline
    from haystack.tools import ComponentTool
    from haystack.components.websearch import SerperDevWebSearch
    from haystack.utils import Secret
    from haystack.components.tools.tool_invoker import ToolInvoker
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    # Create a SerperDev search component
    search = SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"), top_k=3)

    # Create a tool from the component
    tool = ComponentTool(
        component=search,
        name="web_search",  # Optional: defaults to "serper_dev_web_search"
        description="Search the web for current information on any topic"  # Optional: defaults to component docstring
    )

    # Create pipeline with OpenAIChatGenerator and ToolInvoker
    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini", tools=[tool]))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

    # Connect components
    pipeline.connect("llm.replies", "tool_invoker.messages")

    message = ChatMessage.from_user("Use the web search tool to find information about Nikola Tesla")

    # Run pipeline
    result = pipeline.run({"llm": {"messages": [message]}})

    print(result)
    ```

    """

    def __init__(
        self,
        component: Component,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        outputs_to_string: Optional[Dict[str, Union[str, Callable[[Any], str]]]] = None,
        inputs_from_state: Optional[Dict[str, str]] = None,
        outputs_to_state: Optional[Dict[str, Dict[str, Union[str, Callable]]]] = None,
    ):
        """
        Create a Tool instance from a Haystack component.

        :param component: The Haystack component to wrap as a tool.
        :param name: Optional name for the tool (defaults to snake_case of component class name).
        :param description: Optional description (defaults to component's docstring).
        :param parameters:
            A JSON schema defining the parameters expected by the Tool.
            Will fall back to the parameters defined in the component's run method signature if not provided.
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
        :raises ValueError: If the component is invalid or schema generation fails.
        """
        if not isinstance(component, Component):
            message = (
                f"Object {component!r} is not a Haystack component. "
                "Use ComponentTool only with Haystack component instances."
            )
            raise ValueError(message)

        if getattr(component, "__haystack_added_to_pipeline__", None):
            msg = (
                "Component has been added to a pipeline and can't be used to create a ComponentTool. "
                "Create ComponentTool from a non-pipeline component instead."
            )
            raise ValueError(msg)

        self._unresolved_parameters = parameters
        # Create the tools schema from the component run method parameters
        tool_schema = parameters or self._create_tool_parameters_schema(component, inputs_from_state or {})

        def component_invoker(**kwargs):
            """
            Invokes the component using keyword arguments provided by the LLM function calling/tool-generated response.

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

        description = description or component.__doc__ or name

        # Create the Tool instance with the component invoker as the function to be called and the schema
        super().__init__(
            name=name,
            description=description,
            parameters=tool_schema,
            function=component_invoker,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
            outputs_to_string=outputs_to_string,
        )
        self._component = component

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the ComponentTool to a dictionary.
        """
        serialized_component = component_to_dict(obj=self._component, name=self.name)

        serialized = {
            "component": serialized_component,
            "name": self.name,
            "description": self.description,
            "parameters": self._unresolved_parameters,
            "outputs_to_string": self.outputs_to_string,
            "inputs_from_state": self.inputs_from_state,
            "outputs_to_state": self.outputs_to_state,
        }

        if self.outputs_to_state is not None:
            serialized_outputs = {}
            for key, config in self.outputs_to_state.items():
                serialized_config = config.copy()
                if "handler" in config:
                    serialized_config["handler"] = serialize_callable(config["handler"])
                serialized_outputs[key] = serialized_config
            serialized["outputs_to_state"] = serialized_outputs

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            serialized["outputs_to_string"] = serialize_callable(self.outputs_to_string["handler"])

        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the ComponentTool from a dictionary.
        """
        inner_data = data["data"]
        component_class = import_class_by_name(inner_data["component"]["type"])
        component = component_from_dict(cls=component_class, data=inner_data["component"], name=inner_data["name"])

        if "outputs_to_state" in inner_data and inner_data["outputs_to_state"]:
            deserialized_outputs = {}
            for key, config in inner_data["outputs_to_state"].items():
                deserialized_config = config.copy()
                if "handler" in config:
                    deserialized_config["handler"] = deserialize_callable(config["handler"])
                deserialized_outputs[key] = deserialized_config
            inner_data["outputs_to_state"] = deserialized_outputs

        if (
            inner_data.get("outputs_to_string") is not None
            and inner_data["outputs_to_string"].get("handler") is not None
        ):
            inner_data["outputs_to_string"]["handler"] = deserialize_callable(
                inner_data["outputs_to_string"]["handler"]
            )

        return cls(
            component=component,
            name=inner_data["name"],
            description=inner_data["description"],
            parameters=inner_data.get("parameters", None),
            outputs_to_string=inner_data.get("outputs_to_string", None),
            inputs_from_state=inner_data.get("inputs_from_state", None),
            outputs_to_state=inner_data.get("outputs_to_state", None),
        )

    def _create_tool_parameters_schema(self, component: Component, inputs_from_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates an OpenAI tools schema from a component's run method parameters.

        :param component: The component to create the schema from.
        :raises SchemaGenerationError: If schema generation fails
        :returns: OpenAI tools schema for the component's run method parameters.
        """
        properties = {}
        required = []

        param_descriptions = self._get_param_descriptions(component.run)

        for input_name, socket in component.__haystack_input__._sockets_dict.items():  # type: ignore[attr-defined]
            if inputs_from_state is not None and input_name in inputs_from_state:
                continue
            input_type = socket.type
            description = param_descriptions.get(input_name, f"Input '{input_name}' for the component.")

            try:
                property_schema = self._create_property_schema(input_type, description)
            except Exception as e:
                raise SchemaGenerationError(
                    f"Error processing input '{input_name}': {e}. "
                    f"Schema generation supports basic types (str, int, float, bool, dict), dataclasses, "
                    f"and lists of these types as input types for component's run method."
                ) from e

            properties[input_name] = property_schema

            # Use socket.is_mandatory to check if the input is required
            if socket.is_mandatory:
                required.append(input_name)

        parameters_schema = {"type": "object", "properties": properties}

        if required:
            parameters_schema["required"] = required

        return parameters_schema

    @staticmethod
    def _get_param_descriptions(method: Callable) -> Dict[str, str]:
        """
        Extracts parameter descriptions from the method's docstring using docstring_parser.

        :param method: The method to extract parameter descriptions from.
        :returns: A dictionary mapping parameter names to their descriptions.
        """
        docstring = getdoc(method)
        if not docstring:
            return {}

        docstring_parser_import.check()
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

    @staticmethod
    def _is_nullable_type(python_type: Any) -> bool:
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

    @staticmethod
    def _create_basic_type_schema(python_type: Any, description: str) -> Dict[str, Any]:
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
        :raises SchemaGenerationError: If schema generation fails, e.g., for unsupported types like Pydantic v2 models
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
        elif hasattr(python_type, "model_validate"):
            raise SchemaGenerationError(
                f"Pydantic models (e.g. {python_type.__name__}) are not supported as input types for "
                f"component's run method."
            )
        else:
            schema = self._create_basic_type_schema(python_type, description)

        if default is not None:
            schema["default"] = default

        return schema

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "ComponentTool":
        # Jinja2 templates throw an Exception when we deepcopy them (see https://github.com/pallets/jinja/issues/758)
        # When we use a ComponentTool in a pipeline at runtime, we deepcopy the tool
        # We overwrite ComponentTool.__deepcopy__ to fix this until a more comprehensive fix is merged.
        # We track the issue here: https://github.com/deepset-ai/haystack/issues/9011
        result = copy(self)

        # Add the object to the memo dictionary to handle circular references
        memo[id(self)] = result

        # Deep copy all attributes with exception handling
        for key, value in self.__dict__.items():
            try:
                # Try to deep copy the attribute
                setattr(result, key, deepcopy(value, memo))
            except TypeError:
                # Fall back to using the original attribute for components that use Jinja2-templates
                logger.debug(
                    "deepcopy of ComponentTool {tool_name} failed. Using original attribute '{attribute}' instead.",
                    tool_name=self.name,
                    attribute=key,
                )
                setattr(result, key, getattr(self, key))

        return result
