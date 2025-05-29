# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin

from pydantic import Field, TypeAdapter, create_model

from haystack import logging
from haystack.core.component import Component
from haystack.core.serialization import (
    component_from_dict,
    component_to_dict,
    generate_qualified_class_name,
    import_class_by_name,
)
from haystack.tools import Tool
from haystack.tools.errors import SchemaGenerationError
from haystack.tools.from_function import _remove_title_from_schema
from haystack.tools.parameters_schema_utils import _get_component_param_descriptions, _resolve_type
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

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
            input_sockets = component.__haystack_input__._sockets_dict  # type: ignore[attr-defined]
            for param_name, param_value in kwargs.items():
                param_type = input_sockets[param_name].type

                # Check if the type (or list element type) has from_dict
                target_type = get_args(param_type)[0] if get_origin(param_type) is list else param_type
                if hasattr(target_type, "from_dict"):
                    if isinstance(param_value, list):
                        resolved_param_value = [
                            target_type.from_dict(item) if isinstance(item, dict) else item for item in param_value
                        ]
                    elif isinstance(param_value, dict):
                        resolved_param_value = target_type.from_dict(param_value)
                    else:
                        resolved_param_value = param_value
                else:
                    # Let TypeAdapter handle both single values and lists
                    type_adapter = TypeAdapter(param_type)
                    resolved_param_value = type_adapter.validate_python(param_value)

                converted_kwargs[param_name] = resolved_param_value
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
        component_run_description, param_descriptions = _get_component_param_descriptions(component)

        # collect fields (types and defaults) and descriptions from function parameters
        fields: Dict[str, Any] = {}

        for input_name, socket in component.__haystack_input__._sockets_dict.items():  # type: ignore[attr-defined]
            if inputs_from_state is not None and input_name in inputs_from_state:
                continue
            input_type = socket.type
            description = param_descriptions.get(input_name, f"Input '{input_name}' for the component.")

            # if the parameter has not a default value, Pydantic requires an Ellipsis (...)
            # to explicitly indicate that the parameter is required
            default = ... if socket.is_mandatory else socket.default_value
            resolved_type = _resolve_type(input_type)
            fields[input_name] = (resolved_type, Field(default=default, description=description))

        try:
            model = create_model(component.run.__name__, __doc__=component_run_description, **fields)
            parameters_schema = model.model_json_schema()
        except Exception as e:
            raise SchemaGenerationError(
                f"Failed to create JSON schema for the run method of Component '{component.__class__.__name__}'"
            ) from e

        # we don't want to include title keywords in the schema, as they contain redundant information
        # there is no programmatic way to prevent Pydantic from adding them, so we remove them later
        # see https://github.com/pydantic/pydantic/discussions/8504
        _remove_title_from_schema(parameters_schema)

        return parameters_schema
