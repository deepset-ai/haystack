from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import os
import sys
import json
import inspect
import logging
from pathlib import Path

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
from haystack.errors import PipelineSchemaError
from haystack.nodes.base import BaseComponent


logger = logging.getLogger(__name__)


JSON_SCHEMAS_PATH = Path(__file__).parent.parent.parent / "haystack" / "json-schemas"
SCHEMA_URL = "https://raw.githubusercontent.com/deepset-ai/haystack-json-schema/main/json-schema/"

# Allows accessory classes (like enums and helpers) to be registered as valid input for
# custom node's init parameters. For now we disable this feature, but flipping this variables
# re-enables it. Mind that string validation will still cut out most attempts to load anything
# else than enums and class constants: see Pipeline.load_from_config()
ALLOW_ACCESSORY_CLASSES = False


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
            name=param.name, kind=param.kind, default=param.default, annotation=get_typed_annotation(param, globalns)
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


def is_valid_component_class(class_):
    return inspect.isclass(class_) and not inspect.isabstract(class_) and issubclass(class_, BaseComponent)


def find_subclasses_in_modules(importable_modules: List[str]):
    """
    This function returns a list `(module, class)` of all the classes that can be imported
    dynamically, for example from a pipeline YAML definition or to generate documentation.

    By default it won't include Base classes, which should be abstract.
    """
    return [
        (module, class_)
        for module in importable_modules
        for _, class_ in inspect.getmembers(sys.modules[module])
        if is_valid_component_class(class_)
    ]


def handle_optional_params(param_fields: List[inspect.Parameter], params_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pydantic v1 cannot generate correct JSON schemas including Optional fields.
    (https://github.com/samuelcolvin/pydantic/issues/1270)
    This function detects optional parameters and updates the schema,
    to allow null values for these parameters.
    To be removed when Pydantic v2 is released and adopted
    """
    optional_params = []
    for param in param_fields:
        is_param_optional = (
            hasattr(param.annotation, "__origin__")
            and param.annotation.__origin__ == Union
            and type(None) in param.annotation.__args__
        )
        if is_param_optional:
            optional_params.append(param)

    for param in optional_params:
        param_dict = params_schema["properties"][param.name]
        type_ = param_dict.pop("type", None)
        if type_ is not None:
            if "items" in param_dict:
                items = param_dict.pop("items")
                param_dict["anyOf"] = [{"type": type_, "items": items}, {"type": "null"}]
            else:
                param_dict["anyOf"] = [{"type": type_}, {"type": "null"}]
        else:
            anyof_list = param_dict.pop("anyOf", None)
            if anyof_list is not None:
                anyof_list = list(sorted(anyof_list, key=lambda x: x["type"]))
                anyof_list.append({"type": "null"})
                param_dict["anyOf"] = anyof_list
    return params_schema


def create_schema_for_node_class(node_class: Type[BaseComponent]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create the JSON schema for a single BaseComponent subclass,
    including all accessory classes.

    :returns: the schema for the node and all accessory classes,
              and a dict with the reference to the node only.
    """
    if not hasattr(node_class, "__name__"):
        raise PipelineSchemaError(
            f"Node class '{node_class}' has no '__name__' attribute, cannot create a schema for it."
        )

    node_name = getattr(node_class, "__name__")

    logger.debug("Creating schema for '%s'", node_name)

    # Read the relevant init parameters from __init__'s signature
    init_method = getattr(node_class, "__init__", None)
    if not init_method:
        raise PipelineSchemaError(f"Could not read the __init__ method of {node_name} to create its schema.")

    signature = get_typed_signature(init_method)

    # Check for variadic parameters (*args or **kwargs) and raise an exception if found
    if any(param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD} for param in signature.parameters.values()):
        raise PipelineSchemaError(
            "Nodes cannot use variadic parameters like *args or **kwargs in their __init__ function."
        )

    param_fields = [
        param for param in signature.parameters.values() if param.kind not in {param.VAR_POSITIONAL, param.VAR_KEYWORD}
    ]
    # Remove self parameter
    param_fields.pop(0)
    param_fields_kwargs: Dict[str, Any] = {}

    # Read all the paramteres extracted from the __init__ method with type and default value
    for param in param_fields:
        annotation = Any
        if param.annotation != param.empty:
            annotation = param.annotation
        default = Required
        if param.default != param.empty:
            default = param.default
        param_fields_kwargs[param.name] = (annotation, default)

    # Create the model with Pydantic and extract the schema
    model = create_model(f"{node_name}ComponentParams", __config__=Config, **param_fields_kwargs)
    model.update_forward_refs(**model.__dict__)
    params_schema = model.schema()

    # Pydantic v1 patch to generate JSON schemas including Optional fields
    # to be removed when Pydantic v2 is released and adopted
    params_schema = handle_optional_params(param_fields, params_schema)

    params_schema["title"] = "Parameters"
    desc = "Each parameter can reference other components defined in the same YAML file."
    params_schema["description"] = desc

    # Definitions for accessory classes will show up here
    params_definitions = {}
    if "definitions" in params_schema:
        if ALLOW_ACCESSORY_CLASSES:
            params_definitions = params_schema.pop("definitions")
        else:
            raise PipelineSchemaError(
                f"Node {node_name} takes object instances as parameters "
                "in its __init__ function. This is currently not allowed: "
                "please use only Python primitives"
            )

    # Write out the schema and ref and return them
    component_name = f"{node_name}Component"
    component_schema = {
        component_name: {
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
                    "const": f"{node_name}",
                },
                "params": params_schema,
            },
            "required": ["type", "name"],
            "additionalProperties": False,
        },
        **params_definitions,
    }
    return component_schema, {"$ref": f"#/definitions/{component_name}"}


def get_json_schema(filename: str, version: str, modules: Optional[List[str]] = None):
    """
    Generate JSON schema for Haystack pipelines.
    """
    if modules is None:
        modules = ["haystack.document_stores", "haystack.nodes"]
    schema_definitions = {}  # All the schemas for the node and accessory classes
    node_refs = []  # References to the nodes only (accessory classes cannot be listed among the nodes in a config)

    # List all known nodes in the given modules
    possible_node_classes = find_subclasses_in_modules(importable_modules=modules)

    # Build the definitions and refs for the nodes
    for _, node_class in possible_node_classes:
        node_definition, node_ref = create_schema_for_node_class(node_class)
        schema_definitions.update(node_definition)
        node_refs.append(node_ref)

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
                "const": version,
            },
            "extras": {
                "title": "Additional properties group",
                "description": "To be specified only if contains special pipelines (for example, if this is a Ray pipeline)",
                "type": "string",
                "enum": ["ray"],
            },
            "components": {
                "title": "Components",
                "description": "Component nodes and their configurations, to later be used in the pipelines section. Define here all the building blocks for the pipelines.",
                "type": "array",
                "items": {"anyOf": node_refs},
                "required": ["type", "name"],
                "additionalProperties": True,  # To allow for custom components in IDEs - will be set to False at validation time.
            },
            "pipelines": {
                "title": "Pipelines",
                "description": "Multiple pipelines can be defined using the components from the same YAML file.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"title": "Name", "description": "Name of the pipeline.", "type": "string"},
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
                                    "serve_deployment_kwargs": {
                                        "title": "serve_deployment_kwargs",
                                        "description": "Arguments to be passed to the Ray Serve `deployment()` method (only for Ray pipelines)",
                                        "type": "object",
                                        "properties": {
                                            "num_replicas": {
                                                "description": "How many replicas Ray should create for this node (only for Ray pipelines)",
                                                "type": "integer",
                                            },
                                            "version": {"type": "string"},
                                            "prev_version": {"type": "string"},
                                            "init_args": {"type": "array"},
                                            "init_kwargs": {"type": "object"},
                                            "router_prefix": {"type": "string"},
                                            "ray_actor_options": {"type": "object"},
                                            "user_config": {"type": {}},
                                            "max_concurrent_queries": {"type": "integer"},
                                        },
                                        "additionalProperties": True,
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
        "oneOf": [
            {
                "not": {"required": ["extras"]},
                "properties": {
                    "pipelines": {
                        "title": "Pipelines",
                        "items": {
                            "properties": {"nodes": {"items": {"not": {"required": ["serve_deployment_kwargs"]}}}}
                        },
                    }
                },
            },
            {"properties": {"extras": {"enum": ["ray"]}}, "required": ["extras"]},
        ],
        "definitions": schema_definitions,
    }

    return pipeline_schema


def inject_definition_in_schema(node_class: Type[BaseComponent], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a node and a schema in dict form, injects the JSON schema for the new component
    so that pipelines containing such note can be validated against it.

    :returns: the updated schema
    """
    if not is_valid_component_class(node_class):
        raise PipelineSchemaError(
            f"Can't generate a valid schema for node of type '{node_class.__name__}'. "
            "Possible causes: \n"
            "   - it has abstract methods\n"
            "   - its __init__() take something else than Python primitive types or other nodes as parameter.\n"
        )
    schema_definition, node_ref = create_schema_for_node_class(node_class)
    schema["definitions"].update(schema_definition)
    schema["properties"]["components"]["items"]["anyOf"].append(node_ref)
    logger.info("Added definition for %s", getattr(node_class, "__name__"))
    return schema


def load_schema():
    """
    Generate the json schema if it doesn't exist and load it
    """
    schema_file_path = JSON_SCHEMAS_PATH / "haystack-pipeline-main.schema.json"
    if not os.path.exists(schema_file_path):
        logging.info("Json schema not found, generating one at: %s", schema_file_path)
        try:
            update_json_schema(main_only=True)
        except Exception as e:
            # Be sure not to remain with an empty file if something went wrong
            if schema_file_path.exists():
                schema_file_path.unlink()
            # This error is not recoverable
            raise e

    with open(schema_file_path, "r") as schema_file:
        return json.load(schema_file)


def update_json_schema(destination_path: Path = JSON_SCHEMAS_PATH, main_only: bool = False):
    """
    Create (or update) a new schema.
    """
    # `main` schema is always updated and will contain the same data as the latest
    # commit from `main` or a release branch
    filename = "haystack-pipeline-main.schema.json"

    os.makedirs(destination_path, exist_ok=True)
    with open(destination_path / filename, "w") as json_file:
        json.dump(get_json_schema(filename=filename, version="ignore"), json_file, indent=2)

    if not main_only and "rc" not in haystack_version:
        # Create/update the specific version file too
        filename = f"haystack-pipeline-{haystack_version}.schema.json"
        with open(destination_path / filename, "w") as json_file:
            json.dump(get_json_schema(filename=filename, version=haystack_version), json_file, indent=2)

        # Update the index
        index_name = "haystack-pipeline.schema.json"
        with open(destination_path / index_name, "r") as json_file:
            index = json.load(json_file)
            new_entry = {
                "allOf": [
                    {"properties": {"version": {"const": haystack_version}}},
                    {"$ref": f"{SCHEMA_URL}haystack-pipeline-{haystack_version}.schema.json"},
                ]
            }
            if new_entry not in index["oneOf"]:
                index["oneOf"].append(new_entry)
        with open(destination_path / index_name, "w") as json_file:
            json.dump(obj=index, fp=json_file, indent=2, sort_keys=True)
