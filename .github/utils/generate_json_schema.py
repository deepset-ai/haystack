import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from haystack import __version__
import haystack.document_stores
import haystack.nodes
import pydantic.schema
from fastapi.dependencies.utils import get_typed_signature
from pydantic import BaseConfig, BaseSettings, Required, SecretStr, create_model
from pydantic.fields import ModelField
from pydantic.schema import SkipField, TypeModelOrEnum, TypeModelSet, encode_default
from pydantic.schema import field_singleton_schema as _field_singleton_schema
from pydantic.typing import is_callable_type
from pydantic.utils import lenient_issubclass

schema_version = __version__
filename = f"haystack-pipeline-{schema_version}.schema.json"
destination_path = Path(__file__).parent.parent.parent / "json-schemas" / filename


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


class Config(BaseConfig):
    extra = "forbid"


def get_json_schema():
    schema_definitions = {}
    additional_definitions = {}

    modules_with_nodes = [haystack.nodes, haystack.document_stores]
    possible_nodes = []
    for module in modules_with_nodes:
        for importable_name in dir(module):
            imported = getattr(module, importable_name)
            possible_nodes.append((module, imported))
    # TODO: decide if there's a better way to not include Base classes other than by
    # the prefix "Base" in the name. Maybe it could make sense to have a list of
    # all the valid nodes to include in the main source code and then using that here.
    for module, node in possible_nodes:
        if lenient_issubclass(node, haystack.nodes.BaseComponent) and not node.__name__.startswith("Base"):
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
                    default = Required
                    if param.default != param.empty:
                        default = param.default
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
                "const": schema_version,
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
                                "additionalProperties": False,
                            },
                            "required": ["name", "nodes"],
                            "additionalProperties": False,
                        },
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


def generate_json_schema():
    pipeline_schema = get_json_schema()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(pipeline_schema, indent=2))


def main():
    from github import Github

    generate_json_schema()
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    logging.info(f"Using config: {settings.json()}")
    g = Github(settings.input_token.get_secret_value())
    repo = g.get_repo(settings.github_repository)

    logging.info("Setting up GitHub Actions git user")
    subprocess.run(["git", "config", "user.name", "github-actions"], check=True)
    subprocess.run(["git", "config", "user.email", "github-actions@github.com"], check=True)
    branch_name = "generate-json-schema"
    logging.info(f"Creating a new branch {branch_name}")
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    logging.info("Adding updated file")
    subprocess.run(["git", "add", str(destination_path)], check=True)
    logging.info("Committing updated file")
    message = "⬆ Upgrade JSON Schema file"
    subprocess.run(["git", "commit", "-m", message], check=True)
    logging.info("Pushing branch")
    subprocess.run(["git", "push", "origin", branch_name], check=True)
    logging.info("Creating PR")
    pr = repo.create_pull(title=message, body=message, base="master", head=branch_name)
    logging.info(f"Created PR: {pr.number}")
    logging.info("Finished")


if __name__ == "__main__":
    # If you only want to generate the JSON Schema file without submitting a PR
    # uncomment this line:
    generate_json_schema()

    # and comment this line:
    # main()
