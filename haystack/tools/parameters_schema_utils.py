# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections
from dataclasses import MISSING, fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Sequence, Union, get_args, get_origin

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from haystack import logging
from haystack.dataclasses import ChatMessage

logger = logging.getLogger(__name__)


# Schema placeholder models for Tool and Toolset
# These are used during JSON schema generation to represent non-serializable types
class _ToolSchemaPlaceholder(BaseModel):
    """Placeholder model representing a Tool for JSON schema generation."""

    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of the tool")
    parameters: dict[str, Any] = Field(description="JSON schema of the tool parameters")


class _ToolsetSchemaPlaceholder(BaseModel):
    """Placeholder model representing a Toolset for JSON schema generation."""

    tools: list[_ToolSchemaPlaceholder] = Field(description="List of tools in the toolset")


def _get_param_descriptions(method: Callable) -> tuple[str, dict[str, str]]:
    """
    Extracts parameter descriptions from the method's docstring using docstring_parser.

    :param method: The method to extract parameter descriptions from.
    :returns:
        A tuple including the short description of the method and a dictionary mapping parameter names to their
        descriptions.
    """
    docstring = getdoc(method)
    if not docstring:
        return "", {}

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
    return parsed_doc.short_description or "", param_descriptions


def _get_component_param_descriptions(component: Any) -> tuple[str, dict[str, str]]:
    """
    Get parameter descriptions from a component, handling both regular Components and SuperComponents.

    For regular components, this extracts descriptions from the run method's docstring.
    For SuperComponents, this extracts descriptions from the underlying pipeline components.

    :param component: The component to extract parameter descriptions from
    :returns: A tuple of (short_description, param_descriptions)
    """
    from haystack.core.super_component.super_component import _SuperComponent

    # Get descriptions from the component's run method
    short_desc, param_descriptions = _get_param_descriptions(component.run)

    # If it's a SuperComponent, enhance the descriptions from the original components
    if isinstance(component, _SuperComponent):
        # Collect descriptions from components in the pipeline
        component_descriptions = []
        processed_components = set()

        # First gather descriptions from all components that have inputs mapped
        for super_param_name, pipeline_paths in component.input_mapping.items():
            # Collect descriptions from all mapped components
            descriptions = []
            for path in pipeline_paths:
                try:
                    # Get the component and socket this input is mapped fromq
                    comp_name, socket_name = component._split_component_path(path)
                    pipeline_component = component.pipeline.get_component(comp_name)

                    # Get run method descriptions for this component
                    run_desc, run_param_descriptions = _get_param_descriptions(pipeline_component.run)

                    # Don't add the same component description multiple times
                    if comp_name not in processed_components:
                        processed_components.add(comp_name)
                        if run_desc:
                            component_descriptions.append(f"'{comp_name}': {run_desc}")

                    # Add parameter description if available
                    if input_param_mapping := run_param_descriptions.get(socket_name):
                        descriptions.append(f"Provided to the '{comp_name}' component as: '{input_param_mapping}'")
                except Exception as e:
                    logger.debug(f"Error extracting description for {super_param_name} from {path}: {str(e)}")

            # We don't only handle a one to one description mapping of input parameters, but a one to many mapping.
            # i.e. for a combined_input parameter description:
            # super_comp = SuperComponent(
            #   pipeline=pipeline,
            #   input_mapping={"combined_input": ["comp_a.query", "comp_b.text"]},
            # )
            if descriptions:
                param_descriptions[super_param_name] = ", and ".join(descriptions) + "."

        # We also create a combined description for the SuperComponent based on its components
        if component_descriptions:
            short_desc = f"A component that combines: {', '.join(component_descriptions)}"

    return short_desc, param_descriptions


def _dataclass_to_pydantic_model(dc_type: Any) -> type[BaseModel]:
    """
    Convert a Python dataclass to an equivalent Pydantic model.

    :param dc_type: The dataclass type to convert.
    :returns:
        A dynamically generated Pydantic model class with fields and types derived from the dataclass definition.
        Field descriptions are extracted from docstrings when available.
    """
    _, param_descriptions = _get_param_descriptions(dc_type)
    cls = dc_type if isinstance(dc_type, type) else dc_type.__class__

    field_defs: dict[str, Any] = {}
    for field in fields(dc_type):
        f_type = field.type if isinstance(field.type, str) else _resolve_type(field.type)
        default = field.default if field.default is not MISSING else ...
        default = field.default_factory() if callable(field.default_factory) else default

        # Special handling for ChatMessage since pydantic doesn't allow for field names with leading underscores
        field_name = field.name
        if dc_type is ChatMessage and field_name.startswith("_"):
            # We remove the underscore since ChatMessage.from_dict does allow for field names without the underscore
            field_name = field_name[1:]

        description = param_descriptions.get(field_name, f"Field '{field_name}' of '{cls.__name__}'.")
        field_defs[field_name] = (f_type, Field(default, description=description))

    model = create_model(cls.__name__, **field_defs)
    return model


def _resolve_type(_type: Any) -> Any:  # noqa: PLR0911  # pylint: disable=too-many-return-statements
    """
    Recursively resolve and convert complex type annotations, transforming dataclasses into Pydantic-compatible types.

    This function walks through nested type annotations (e.g., List, Dict, Union) and converts any dataclass types
    it encounters into corresponding Pydantic models.

    :param _type: The type annotation to resolve. If the type is a dataclass, it will be converted to a Pydantic model.
        For generic types (like list[SomeDataclass]), the inner types are also resolved recursively.

    :returns:
        A fully resolved type, with all dataclass types converted to Pydantic models
    """
    # Special handling for Tool and Toolset types - replace with schema placeholders
    # These types contain Callables which cannot be serialized to JSON Schema
    from haystack.tools.tool import Tool
    from haystack.tools.toolset import Toolset

    if _type is Tool:
        return _ToolSchemaPlaceholder

    if _type is Toolset:
        return _ToolsetSchemaPlaceholder

    if is_dataclass(_type):
        return _dataclass_to_pydantic_model(_type)

    origin = get_origin(_type)
    args = get_args(_type)

    if origin is list:
        return list[_resolve_type(args[0]) if args else Any]  # type: ignore[misc]

    if origin is collections.abc.Sequence:
        return Sequence[_resolve_type(args[0]) if args else Any]  # type: ignore[misc]

    if origin is Union:
        return Union[tuple(_resolve_type(a) for a in args)]

    if origin is dict:
        return dict[args[0] if args else Any, _resolve_type(args[1]) if args else Any]  # type: ignore[misc]

    return _type
