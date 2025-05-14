# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import collections
from dataclasses import MISSING, fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, get_args, get_origin

from pydantic import Field, create_model

from haystack import logging
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools.errors import SchemaGenerationError
from haystack.tools.from_function import _remove_title_from_schema

with LazyImport(message="Run 'pip install docstring-parser'") as docstring_parser_import:
    from docstring_parser import parse


logger = logging.getLogger(__name__)


def _get_param_descriptions(method: Callable) -> Tuple[str, Dict[str, str]]:
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
    return parsed_doc.short_description or "", param_descriptions


def _dataclass_to_pydantic_model(dc_type: Any) -> Any:
    _, param_descriptions = _get_param_descriptions(dc_type)
    cls = dc_type if isinstance(dc_type, type) else dc_type.__class__

    field_defs: Dict[str, Any] = {}
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


def _resolve_type(_type: Any) -> Any:
    if is_dataclass(_type):
        return _dataclass_to_pydantic_model(_type)

    origin = get_origin(_type)
    args = get_args(_type)

    if origin is list:
        return List[_resolve_type(args[0]) if args else Any]  # type: ignore

    if origin is collections.abc.Sequence:
        return Sequence[_resolve_type(args[0]) if args else Any]  # type: ignore

    if origin is Union:
        return Union[tuple(_resolve_type(a) for a in args)]  # type: ignore

    if origin is dict:
        return Dict[args[0] if args else Any, _resolve_type(args[1]) if args else Any]  # type: ignore

    return _type


def _create_parameters_schema(function_name: str, description: Optional[str], field_definitions: Any):
    try:
        model = create_model(function_name, __doc__=description, **field_definitions)
        parameters_schema = model.model_json_schema()
    except Exception as e:
        raise SchemaGenerationError(f"Failed to create JSON schema for function '{function_name}'") from e

    # we don't want to include title keywords in the schema, as they contain redundant information
    # there is no programmatic way to prevent Pydantic from adding them, so we remove them later
    # see https://github.com/pydantic/pydantic/discussions/8504
    _remove_title_from_schema(parameters_schema)

    return parameters_schema
