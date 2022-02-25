from typing import Any, Callable, Dict, Optional, Set, Tuple

import json
import inspect
from pathlib import Path
import logging

from haystack.pipelines.base import JSON_SCHEMAS_PATH

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
    field_singleton_schema as _field_singleton_schema
)
 
from haystack import __version__ as haystack_version
from haystack.nodes.base import BaseComponent



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
    extra = "forbid"


def get_json_schema(filename: str):
    """
    Generate JSON schema for Haystack pipelines.
    """
    schema_definitions = {}
    additional_definitions = {}

    possible_nodes = BaseComponent._find_subclasses_in_modules()
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
                logging.info(f"--- processing param: {param.name}")
                annotation = Any
                if param.annotation != param.empty:
                    annotation = param.annotation
                    logging.info(f"       annotation: {annotation}")
                default = Required
                if param.default != param.empty:
                    default = param.default
                    logging.info(f"       default: {default}")
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
    pipeline_schema = {
        "$schema": "http://json-schema.org/draft-07/schema",
        "$id": f"https://haystack.deepset.ai/json-schemas/{filename}",
        "title": "Haystack Pipeline",
        "description": "Haystack Pipeline YAML file describing the nodes of the pipelines. For more info read the docs at: https://haystack.deepset.ai/components/pipelines#yaml-file-definitions",
        "type": "object",
        "properties": {
            "version": {
                "title": "Version",
                "description": "Version of the Haystack Pipeline file.",
                "type": "string",
                "const": haystack_version,
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
                        "type": {
                            "title": "Type",
                            "description": "Type of pipeline (Query, Indexing, or custom types).",
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


def list_indexed_versions(index):
    """
    Given the schema index as a parsed JSON,
    return a list of all the versions it contains.
    """
    indexed_versions = []
    for version_entry in index["oneOf"]:
        for property_entry in version_entry["allOf"]:
            if "properties" in property_entry.keys():
                indexed_versions.append(property_entry["properties"]["version"]["const"])
    return indexed_versions


def cleanup_rc_versions(index):
    """
    Given the schema index as a parsed JSON,
    removes any existing (unstable) rc version from it.
    """
    new_versions_list = []
    for version_entry in index["oneOf"]:
        for property_entry in version_entry["allOf"]:
            if "properties" in property_entry.keys():
                if "rc" not in property_entry["properties"]["version"]["const"]:
                    new_versions_list.append(version_entry)
                    break
    index["oneOf"] = new_versions_list
    return index


def new_version_entry(version):
    """
    Returns a new entry for the version index JSON schema.
    """
    return {
        "allOf": [
            {"properties": {"version": {"const": version}}},
            {
                "$ref": "https://raw.githubusercontent.com/deepset-ai/haystack/master/json-schemas/"
                f"haystack-pipeline-{version}.schema.json"
            },
        ]
    }


def generate_json_schema(
    update_index: bool, 
    filename: str = f"haystack-pipeline-{haystack_version}.schema.json",
    destination_path: Path = JSON_SCHEMAS_PATH,
    index_path: Path = JSON_SCHEMAS_PATH / "haystack-pipeline.schema.json"
):
    # Create new schema file
    pipeline_schema = get_json_schema(filename)
    destination_path.mkdir(parents=True, exist_ok=True)
    (destination_path / filename).write_text(json.dumps(pipeline_schema, indent=2))

    # Update schema index
    if update_index:
        index = []
        with open(index_path, "r") as index_file:
            index = json.load(index_file)
        if index:
            index = cleanup_rc_versions(index)
            indexed_versions = list_indexed_versions(index)
            if not any(version == haystack_version for version in indexed_versions):
                index["oneOf"].append(new_version_entry(haystack_version))
                with open(index_path, "w") as index_file:
                    json.dump(index, index_file, indent=4)


if __name__ == "__main__":
    generate_json_schema(update_index=True)
