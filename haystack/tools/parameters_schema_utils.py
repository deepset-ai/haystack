# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections
import keyword
import types
from collections.abc import Callable, Sequence
from collections.abc import Callable as ABCCallable
from dataclasses import MISSING, fields, is_dataclass
from inspect import getdoc
from types import NoneType
from typing import Annotated, Any, Union, get_args, get_origin

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage
from haystack.utils.type_serialization import _is_union_type

logger = logging.getLogger(__name__)


def _unwrap_optional(type_hint: Any) -> Any:
    """
    Unwrap Optional types (i.e. ``X | None`` or ``Optional[X]``) to get the inner type.

    :param type_hint: The type hint to unwrap.
    :returns: The inner type if ``type_hint`` is ``Optional[X]``, otherwise ``type_hint`` unchanged.
    """
    origin = get_origin(type_hint)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(type_hint) if a is not NoneType]
        if len(non_none) == 1:
            return non_none[0]
    return type_hint


def _is_state_annotation(annotation: Any) -> bool:
    """Recognize State through Annotated and optional wrappers."""
    if get_origin(annotation) is Annotated:
        return _is_state_annotation(get_args(annotation)[0])

    unwrapped = _unwrap_optional(annotation)
    if unwrapped is not annotation:
        return _is_state_annotation(unwrapped)

    return annotation is State


def _contains_callable_type(type_hint: Any) -> bool:
    """
    Check if a type hint contains a Callable type, including within Union types.

    The purpose of this function is to help identify Callable types so they can
    be skipped during schema generation.

    :param type_hint: The type hint to check.
    :returns: True if the type contains a Callable, False otherwise.
    """
    origin = get_origin(type_hint)

    # Check if it's a Callable type (direct or parameterized)
    if type_hint in (Callable, ABCCallable) or origin in (Callable, ABCCallable):
        return True

    # Recursively check Union types (both typing.Union and types.UnionType for `X | Y` syntax)
    if origin in (Union, types.UnionType):
        return any(_contains_callable_type(arg) for arg in get_args(type_hint))

    return False


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
                "Missing description for parameter '{arg_name}'. Please add a description in the component's "
                "run() method docstring using the format ':param {arg_name}: <description>'. "
                "This description helps the LLM understand how to use this parameter.",
                arg_name=param.arg_name,
            )
        param_descriptions[param.arg_name] = param.description.strip() if param.description else ""
    return parsed_doc.short_description or "", param_descriptions


def _get_component_param_descriptions(component: Any) -> dict[str, str]:
    """
    Get parameter descriptions from a component, handling both regular Components and SuperComponents.

    For regular components, this extracts descriptions from the run method's docstring.
    For SuperComponents, this extracts descriptions from the underlying pipeline components.

    :param component: The component to extract parameter descriptions from
    :returns: A dictionary mapping parameter names to their descriptions
    """
    from haystack.core.super_component.super_component import _SuperComponent

    # Get descriptions from the component's run method
    _, param_descriptions = _get_param_descriptions(component.run)

    # If it's a SuperComponent, enhance the parameter descriptions from the original components
    if isinstance(component, _SuperComponent):
        for super_param_name, pipeline_paths in component.input_mapping.items():
            # Collect descriptions from all mapped components
            descriptions = []
            for path in pipeline_paths:
                try:
                    # Get the component and socket this input is mapped from
                    comp_name, socket_name = component._split_component_path(path)
                    pipeline_component = component.pipeline.get_component(comp_name)

                    # Add parameter description if available
                    _, run_param_descriptions = _get_param_descriptions(pipeline_component.run)
                    if input_param_mapping := run_param_descriptions.get(socket_name):
                        descriptions.append(f"Provided to the '{comp_name}' component as: '{input_param_mapping}'")
                except Exception as e:
                    logger.debug(
                        "Error extracting description for {super_param_name} from {path}: {e}",
                        super_param_name=super_param_name,
                        path=path,
                        e=str(e),
                    )

            # A single SuperComponent input can map to multiple pipeline components, e.g.
            # input_mapping={"combined_input": ["comp_a.query", "comp_b.text"]}
            if descriptions:
                param_descriptions[super_param_name] = ", and ".join(descriptions) + "."

    return param_descriptions


def _safe_internal_field_name(param_name: str, taken: set[str]) -> str:
    """
    Return a Pydantic field name that does not start with an underscore.

    ``create_model`` treats a leading underscore as a private attribute. The returned name is
    internal only; callers set ``Field(alias=param_name)`` so the JSON schema uses ``param_name``.

    :param param_name: The real parameter name, which starts with ``_``.
    :param taken: Field names already reserved. Not modified.
    :returns: A unique identifier that does not start with ``_``.
    """
    base = param_name.lstrip("_")
    if not base.isidentifier() or keyword.iskeyword(base):
        base = "param"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _create_model_fields(
    specs: list[tuple[str, Any, Any]], *, descriptions: dict[str, str] | None = None
) -> dict[str, Any]:
    """
    Build ``create_model`` field definitions for tool parameters.

    Pydantic drops names that start with ``_`` from ``model_json_schema()`` when they are passed as
    ``(annotation, default)``, and raises if they are passed as ``Field``. Those parameters are stored
    under a safe internal name with ``Field(alias=param_name)`` so the schema still exposes the real
    name, including whether the parameter is required.

    When ``descriptions`` is omitted, parameters that do not start with ``_`` keep the
    ``(annotation, default)`` form. Callers that read descriptions from ``Annotated`` metadata can
    apply them to the schema afterwards. When ``descriptions`` is provided, every field is a ``Field``
    and includes that description.

    :param specs: ``(name, annotation, default)`` triples in schema order. ``default`` is ``...`` when
        the parameter is required.
    :param descriptions: Optional mapping of parameter name to description.
    :returns: A mapping suitable for ``create_model(model_name, **fields)``.
    """
    taken = {name for name, _, _ in specs if not name.startswith("_")}
    field_defs: dict[str, Any] = {}
    for name, annotation, default in specs:
        if name.startswith("_"):
            internal = _safe_internal_field_name(name, taken)
            taken.add(internal)
            if descriptions is None:
                field_defs[internal] = (annotation, Field(default, alias=name))
            else:
                field_defs[internal] = (annotation, Field(default, alias=name, description=descriptions[name]))
            continue
        if descriptions is None:
            field_defs[name] = (annotation, default)
        else:
            field_defs[name] = (annotation, Field(default, description=descriptions[name]))
    return field_defs


def _dataclass_to_pydantic_model(dc_type: Any) -> type[BaseModel]:
    """
    Convert a Python dataclass to an equivalent Pydantic model.

    Field names that start with an underscore are included in the schema under that name.
    ``ChatMessage`` is the exception: its leading underscores are removed so the schema matches
    ``ChatMessage.from_dict``, which accepts ``role``, ``content``, ``name``, and ``meta``.

    :param dc_type: The dataclass type to convert.
    :returns:
        A dynamically generated Pydantic model class with fields and types derived from the dataclass definition.
        Field descriptions are extracted from docstrings when available.
    """
    _, param_descriptions = _get_param_descriptions(dc_type)
    cls = dc_type if isinstance(dc_type, type) else dc_type.__class__

    specs: list[tuple[str, Any, Any]] = []
    descriptions: dict[str, str] = {}
    for field in fields(dc_type):
        f_type = field.type if isinstance(field.type, str) else _resolve_type(field.type)
        default = field.default if field.default is not MISSING else ...
        default = field.default_factory() if callable(field.default_factory) else default

        # ChatMessage.from_dict accepts these names without the leading underscore.
        field_name = field.name
        if dc_type is ChatMessage and field_name.startswith("_"):
            field_name = field_name[1:]

        specs.append((field_name, f_type, default))
        descriptions[field_name] = param_descriptions.get(field_name, f"Field '{field_name}' of '{cls.__name__}'.")

    return create_model(cls.__name__, **_create_model_fields(specs, descriptions=descriptions))


def _resolve_type(_type: Any) -> Any:  # noqa: PLR0911
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

    if _is_union_type(origin):
        return Union[tuple(_resolve_type(a) for a in args)]

    if origin is dict:
        return dict[args[0] if args else Any, _resolve_type(args[1]) if args else Any]  # type: ignore[misc]

    return _type
