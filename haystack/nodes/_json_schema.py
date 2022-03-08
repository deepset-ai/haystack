from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import os
import re
import sys
import json
import inspect
from pathlib import Path
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)

import pydantic.schema
from pydantic import BaseConfig, BaseSettings, Required, SecretStr, create_model
from pydantic.typing import ForwardRef, evaluate_forwardref, is_callable_type
from pydantic.fields import ModelField
from pydantic.schema import (
    SkipField,
    TypeModelOrEnum,
    TypeModelSet,
    encode_default,
    field_singleton_schema as _field_singleton_schema,
)

from haystack import __version__ as haystack_version
from haystack.nodes.base import BaseComponent
from haystack.pipelines.config import JSON_SCHEMAS_PATH


SCHEMA_URL = "https://haystack.deepset.ai/json-schemas/"


class Settings(BaseSettings):
    input_token: SecretStr
    github_repository: str


# Monkey patch Pydantic's field_singleton_schema to convert classes and functions to
# strings in JSON Schema
def field_singleton_schema(
    field: ModelField,
    *,
    by_alias: bool,
    model_name_map: Dict[TypeModelOrEnum, str],
    ref_template: str,
    schema_overrides: bool = False,
    ref_prefix: Optional[str] = None,
    known_models: TypeModelSet,
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    try:
        return _field_singleton_schema(
            field,
            by_alias=by_alias,
            model_name_map=model_name_map,
            ref_template=ref_template,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            known_models=known_models,
        )
    except (ValueError, SkipField):
        schema: Dict[str, Any] = {"type": "string"}

        if isinstance(field.default, type) or is_callable_type(field.default):
            default = field.default.__name__
        else:
            default = field.default
        if not field.required:
            schema["default"] = encode_default(default)
        return schema, {}, set()


# Monkeypatch Pydantic's field_singleton_schema
pydantic.schema.field_singleton_schema = field_singleton_schema


# From FastAPI's internals
def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param, globalns),
        )
        for param in signature.parameters.values()
    ]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature


# From FastAPI's internals
def get_typed_annotation(param: inspect.Parameter, globalns: Dict[str, Any]) -> Any:
    annotation = param.annotation
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation


class Config(BaseConfig):
    extra = "forbid"  # type: ignore


def find_subclasses_in_modules(
    include_base_classes: bool = False, importable_modules=["haystack.document_stores", "haystack.nodes"]
):
    """
    This function returns a list `(module, class)` of all the classes that can be imported
    dynamically, for example from a pipeline YAML definition or to generate documentation.

    By default it won't include Base classes, which should be abstract.
    """
    return [
        (module, clazz)
        for module in importable_modules
        for _, clazz in inspect.getmembers(sys.modules[module])
        if (
            inspect.isclass(clazz)
            and not inspect.isabstract(clazz)
            and issubclass(clazz, BaseComponent)
            and (include_base_classes or not clazz.__name__.startswith("Base"))
        )
    ]


def get_json_schema(filename: str, compatible_versions: List[str]):
    """
    Generate JSON schema for Haystack pipelines.
    """
    schema_definitions = {}
    additional_definitions = {}

    possible_nodes = find_subclasses_in_modules()
    for _, node in possible_nodes:
        logging.info(f"Processing node: {node.__name__}")
        init_method = getattr(node, "__init__", None)
        if init_method:
            signature = get_typed_signature(init_method)
            param_fields = [
                param
                for param in signature.parameters.values()
                if param.kind not in {param.VAR_POSITIONAL, param.VAR_KEYWORD}
            ]
            # Remove self parameter
            param_fields.pop(0)
            param_fields_kwargs: Dict[str, Any] = {}

            for param in param_fields:
                # logging.info(f"--- processing param: {param.name}")
                annotation = Any
                if param.annotation != param.empty:
                    annotation = param.annotation
                    # logging.info(f"       annotation: {annotation}")
                default = Required
                if param.default != param.empty:
                    default = param.default
                    # logging.info(f"       default: {default}")
                param_fields_kwargs[param.name] = (annotation, default)

            model = create_model(
                f"{node.__name__}ComponentParams",
                __config__=Config,
                **param_fields_kwargs,
            )

            model.update_forward_refs(**model.__dict__)
            params_schema = model.schema()
            params_schema["title"] = "Parameters"
            params_schema[
                "description"
            ] = "Each parameter can reference other components defined in the same YAML file."
            if "definitions" in params_schema:
                params_definitions = params_schema.pop("definitions")
                additional_definitions.update(params_definitions)
            component_schema = {
                "type": "object",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "Custom name for the component. Helpful for visualization and debugging.",
                        "type": "string",
                    },
                    "type": {
                        "title": "Type",
                        "description": "Haystack Class name for the component.",
                        "type": "string",
                        "const": f"{node.__name__}",
                    },
                    "params": params_schema,
                },
                "required": ["type", "name"],
                "additionalProperties": False,
            }
            schema_definitions[f"{node.__name__}Component"] = component_schema

    all_definitions = {**schema_definitions, **additional_definitions}
    component_refs = [{"$ref": f"#/definitions/{name}"} for name in schema_definitions]
    compatible_version_strings = [{"const": version} for version in compatible_versions]

    pipeline_schema = {
        "$schema": "http://json-schema.org/draft-07/schema",
        "$id": f"{SCHEMA_URL}{filename}",
        "title": "Haystack Pipeline",
        "description": "Haystack Pipeline YAML file describing the nodes of the pipelines. For more info read the docs at: https://haystack.deepset.ai/components/pipelines#yaml-file-definitions",
        "type": "object",
        "properties": {
            "version": {
                "title": "Version",
                "description": "Version of the Haystack Pipeline file.",
                "type": "string",
                "oneOf": compatible_version_strings,
            },
            "components": {
                "title": "Components",
                "description": "Component nodes and their configurations, to later be used in the pipelines section. Define here all the building blocks for the pipelines.",
                "type": "array",
                "items": {"anyOf": component_refs},
                "required": ["type", "name"],
                "additionalProperties": False,
            },
            "pipelines": {
                "title": "Pipelines",
                "description": "Multiple pipelines can be defined using the components from the same YAML file.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "title": "Name",
                            "description": "Name of the pipeline.",
                            "type": "string",
                        },
                        "nodes": {
                            "title": "Nodes",
                            "description": "Nodes to be used by this particular pipeline",
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "title": "Name",
                                        "description": "The name of this particular node in the pipeline. This should be one of the names from the components defined in the same file.",
                                        "type": "string",
                                    },
                                    "inputs": {
                                        "title": "Inputs",
                                        "description": "Input parameters for this node.",
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["name", "inputs"],
                                "additionalProperties": False,
                            },
                            "required": ["name", "nodes"],
                            "additionalProperties": False,
                        },
                        "additionalProperties": False,
                    },
                    "additionalProperties": False,
                },
            },
        },
        "required": ["version", "components", "pipelines"],
        "additionalProperties": False,
        "definitions": all_definitions,
    }
    return pipeline_schema


def natural_sort(list_to_sort: List[str]) -> List[str]:
    """Sorts a list keeping numbers in the correct numerical order"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanumeric_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(list_to_sort, key=alphanumeric_key)


def load(path: Path) -> Dict[str, Any]:
    """Shorthand for loading a JSON"""
    with open(path, "r") as json_file:
        return json.load(json_file)


def dump(data: Dict[str, Any], path: Path) -> None:
    """Shorthand for dumping to JSON"""
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2)


def new_version_entry(version):
    """
    Returns a new entry for the version index JSON schema.
    """
    return {
        "allOf": [
            {"properties": {"version": {"oneOf": [{"const": version}]}}},
            {
                "$ref": "https://raw.githubusercontent.com/deepset-ai/haystack/master/json-schemas/"
                f"haystack-pipeline-{version}.schema.json"
            },
        ]
    }


def update_json_schema(
    update_index: bool,
    destination_path: Path = JSON_SCHEMAS_PATH,
    index_path: Path = JSON_SCHEMAS_PATH / "haystack-pipeline.schema.json",
):
    # Locate the latest schema's path
    latest_schema_path = destination_path / Path(
        natural_sort(os.listdir(destination_path))[-3]
    )  # -1 is index, -2 is unstable
    logging.info(f"Latest schema: {latest_schema_path}")
    latest_schema = load(latest_schema_path)

    # List the versions supported by the last schema
    supported_versions_block = deepcopy(latest_schema["properties"]["version"]["oneOf"])
    supported_versions = [entry["const"].replace('"', "") for entry in supported_versions_block]
    logging.info(f"Versions supported by this schema: {supported_versions}")

    # Create new schema with the same filename and versions embedded, to be identical to the latest one.
    new_schema = get_json_schema(latest_schema_path.name, supported_versions)

    # If the two schemas are identical, no need to write a new one:
    # Just add the new version to the list of versions supported by
    # the latest schema if it's not there yet
    unstable_versions_block = []
    if json.dumps(new_schema) == json.dumps(latest_schema):
        logging.info("No difference in the schema, won't create a new file.")

        if haystack_version in supported_versions:
            logging.info(
                f"Version {haystack_version} was already supported " f"(supported versions: {supported_versions})"
            )
        else:
            logging.info(
                f"This version ({haystack_version}) was not listed "
                f"(supported versions: {supported_versions}): "
                "updating the supported versions list."
            )

            # Updating the latest schema's list of supported versions
            supported_versions_block.append({"const": haystack_version})
            unstable_versions_block = supported_versions_block
            latest_schema["properties"]["version"]["oneOf"] = supported_versions_block
            dump(latest_schema, latest_schema_path)

            # Update the JSON schema index too
            if update_index:
                index = load(index_path)
                index["oneOf"][-1]["allOf"][0]["properties"]["version"]["oneOf"] = supported_versions_block
                dump(index, index_path)

    # If the two schemas are different, then there has been some
    # changes into the schema, so we need a new file. Update the schema's
    # filename and supported versions, then save it.
    else:
        filename = f"haystack-pipeline-{haystack_version}.schema.json"
        logging.info(f"The schemas are different: adding {filename} to the schema folder.")

        # Let's check if the schema changed without a version change
        if haystack_version in supported_versions and len(supported_versions) > 1:
            logging.info(
                f"Version {haystack_version} was supported by the latest schema"
                f"(supported versions: {supported_versions}). "
                f"Removing support for version {haystack_version} from it."
            )

            supported_versions_block = [
                entry for entry in supported_versions_block if entry["const"].replace('"', "") != haystack_version
            ]
            latest_schema["properties"]["version"]["oneOf"] = supported_versions_block
            dump(latest_schema, latest_schema_path)

            # Update the JSON schema index too
            if update_index:
                index = load(index_path)
                index["oneOf"][-1]["allOf"][0]["properties"]["version"]["oneOf"] = supported_versions_block
                dump(index, index_path)

        # Dump the new schema file
        new_schema["$id"] = f"{SCHEMA_URL}{filename}"
        unstable_versions_block = [{"const": haystack_version}]
        new_schema["properties"]["version"]["oneOf"] = [{"const": haystack_version}]
        dump(new_schema, destination_path / filename)

        # Update schema index with a whole new entry
        if update_index:
            index = load(index_path)
            new_entry = new_version_entry(haystack_version)
            if all(new_entry != entry for entry in index["oneOf"]):
                index["oneOf"].append(new_version_entry(haystack_version))
            dump(index, index_path)

    # Update the unstable schema (for tests and internal use).
    unstable_filename = "haystack-pipeline-unstable.schema.json"
    unstable_schema = deepcopy(new_schema)
    unstable_schema["$id"] = f"{SCHEMA_URL}{unstable_filename}"
    unstable_schema["properties"]["version"]["oneOf"] = [{"const": "unstable"}] + unstable_versions_block
    dump(unstable_schema, destination_path / unstable_filename)
