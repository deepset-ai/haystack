from typing import Any, Dict, List, Optional

import re
import os
import copy
import logging
from pathlib import Path
from networkx import DiGraph
import yaml
import json
from jsonschema.validators import Draft7Validator
from jsonschema.exceptions import ValidationError

from haystack import __version__
from haystack.nodes.base import BaseComponent
from haystack.nodes._json_schema import inject_definition_in_schema, JSON_SCHEMAS_PATH
from haystack.errors import PipelineConfigError, PipelineSchemaError, HaystackError


logger = logging.getLogger(__name__)


VALID_INPUT_REGEX = re.compile(r"^[-a-zA-Z0-9_/.:]+$")


def get_pipeline_definition(pipeline_config: Dict[str, Any], pipeline_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the definition of Pipeline from a given pipeline config. If the config contains more than one Pipeline,
    then the pipeline_name must be supplied.

    :param pipeline_config: Dict Pipeline config parsed as a dictionary.
    :param pipeline_name: name of the Pipeline.
    """
    if pipeline_name is None:
        if len(pipeline_config["pipelines"]) == 1:
            pipeline_definition = pipeline_config["pipelines"][0]
        else:
            raise PipelineConfigError("The YAML contains multiple pipelines. Please specify the pipeline name to load.")
    else:
        pipelines_in_definitions = list(filter(lambda p: p["name"] == pipeline_name, pipeline_config["pipelines"]))
        if not pipelines_in_definitions:
            raise PipelineConfigError(
                f"Cannot find any pipeline with name '{pipeline_name}' declared in the YAML file. "
                f"Existing pipelines: {[p['name'] for p in pipeline_config['pipelines']]}"
            )
        pipeline_definition = pipelines_in_definitions[0]

    return pipeline_definition


def get_component_definitions(pipeline_config: Dict[str, Any], overwrite_with_env_variables: bool) -> Dict[str, Any]:
    """
    Returns the definitions of all components from a given pipeline config.

    :param pipeline_config: Dict Pipeline config parsed as a dictionary.
    :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                            to change index name param for an ElasticsearchDocumentStore, an env
                                            variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                            `_` sign must be used to specify nested hierarchical properties.
    """
    component_definitions = {}  # definitions of each component from the YAML.
    raw_component_definitions = copy.deepcopy(pipeline_config["components"])
    for component_definition in raw_component_definitions:
        if overwrite_with_env_variables:
            _overwrite_with_env_variables(component_definition)
        name = component_definition.pop("name")
        component_definitions[name] = component_definition

    return component_definitions


def read_pipeline_config_from_yaml(path: Path) -> Dict[str, Any]:
    """
    Parses YAML files into Python objects.
    Fails if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_config_strings(pipeline_config: Any):
    """
    Ensures that strings used in the pipelines configuration
    contain only alphanumeric characters and basic punctuation.
    """
    try:
        if isinstance(pipeline_config, dict):
            for key, value in pipeline_config.items():
                validate_config_strings(key)
                validate_config_strings(value)

        elif isinstance(pipeline_config, list):
            for value in pipeline_config:
                validate_config_strings(value)

        else:
            if not VALID_INPUT_REGEX.match(str(pipeline_config)):
                raise PipelineConfigError(
                    f"'{pipeline_config}' is not a valid variable name or value. "
                    "Use alphanumeric characters or dash, underscore and colon only."
                )
    except RecursionError as e:
        raise PipelineConfigError("The given pipeline configuration is recursive, can't validate it.") from e


def build_component_dependency_graph(
    pipeline_definition: Dict[str, Any], component_definitions: Dict[str, Any]
) -> DiGraph:
    """
    Builds a dependency graph between components. Dependencies are:
    - referenced components during component build time (e.g. init params)
    - predecessor components in the pipeline that produce the needed input

    This enables sorting the components in a working and meaningful order for instantiation using topological sorting.

    :param pipeline_definition: the definition of the pipeline (e.g. use get_pipeline_definition() to obtain it)
    :param component_definitions: the definition of the pipeline components (e.g. use get_component_definitions() to obtain it)
    """
    graph = DiGraph()
    for component_name, component_definition in component_definitions.items():
        params = component_definition.get("params", {})
        referenced_components: List[str] = list()
        for param_value in params.values():
            # Currently we don't do any additional type validation here.
            # See https://github.com/deepset-ai/haystack/pull/2253#discussion_r815951591.
            if param_value in component_definitions:
                referenced_components.append(param_value)
        for referenced_component in referenced_components:
            graph.add_edge(referenced_component, component_name)
    for node in pipeline_definition["nodes"]:
        node_name = node["name"]
        graph.add_node(node_name)
        for input in node["inputs"]:
            if input in component_definitions:
                # Special case for (actually permitted) cyclic dependencies between two components:
                # e.g. DensePassageRetriever depends on ElasticsearchDocumentStore.
                # In indexing pipelines ElasticsearchDocumentStore depends on DensePassageRetriever's output.
                # But this second dependency is looser, so we neglect it.
                if not graph.has_edge(node_name, input):
                    graph.add_edge(input, node_name)
    return graph


def validate_yaml(path: Path):
    """
    Validates the given YAML file using the autogenerated JSON schema.

    :param pipeline_config: the configuration to validate
    :return: None if validation is successful
    :raise: `PipelineConfigError` in case of issues.
    """
    pipeline_config = read_pipeline_config_from_yaml(path)
    validate_config(pipeline_config=pipeline_config)
    logging.debug(f"'{path}' contains valid Haystack pipelines.")


def validate_config(pipeline_config: Dict) -> None:
    """
    Validates the given configuration using the autogenerated JSON schema.

    :param pipeline_config: the configuration to validate
    :return: None if validation is successful
    :raise: `PipelineConfigError` in case of issues.
    """
    validate_config_strings(pipeline_config)

    with open(JSON_SCHEMAS_PATH / f"haystack-pipeline-unstable.schema.json", "r") as schema_file:
        schema = json.load(schema_file)

    compatible_versions = [version["const"].replace('"', "") for version in schema["properties"]["version"]["oneOf"]]
    loaded_custom_nodes = []

    while True:

        try:
            Draft7Validator(schema).validate(instance=pipeline_config)

            if pipeline_config["version"] == "unstable":
                logging.warning(
                    "You seem to be using the 'unstable' version of the schema to validate "
                    "your pipeline configuration.\n"
                    "This is NOT RECOMMENDED in production environments, as pipelines "
                    "might manage to load and then misbehave without warnings.\n"
                    f"Please pin your configurations to '{__version__}' to ensure stability."
                )

            elif pipeline_config["version"] not in compatible_versions:
                raise PipelineConfigError(
                    f"Cannot load pipeline configuration of version {pipeline_config['version']} "
                    f"in Haystack version {__version__} "
                    f"(only versions {compatible_versions} are compatible with this Haystack release).\n"
                    "Please check out the release notes (https://github.com/deepset-ai/haystack/releases/latest), "
                    "the documentation (https://haystack.deepset.ai/components/pipelines#yaml-file-definitions) "
                    "and fix your configuration accordingly."
                )
            break

        except ValidationError as validation:

            # If the validation comes from an unknown node, try to find it and retry:
            if list(validation.relative_schema_path) == ["properties", "components", "items", "anyOf"]:
                if validation.instance["type"] not in loaded_custom_nodes:

                    logger.info(
                        f"Missing definition for node of type {validation.instance['type']}. Looking into local classes..."
                    )
                    missing_component_class = BaseComponent.get_subclass(validation.instance["type"])
                    schema = inject_definition_in_schema(node_class=missing_component_class, schema=schema)
                    loaded_custom_nodes.append(validation.instance["type"])
                    continue

                # A node with the given name was in the schema, but something else is wrong with it.
                # Probably it references unknown classes in its init parameters.
                raise PipelineSchemaError(
                    f"Node of type {validation.instance['type']} found, but it failed validation. Possible causes:\n"
                    " - The node is missing some mandatory parameter\n"
                    " - Wrong indentation of some parameter in YAML\n"
                    "See the stacktrace for more information."
                ) from validation

            # Format the error to make it as clear as possible
            error_path = [
                i
                for i in list(validation.relative_schema_path)[:-1]
                if repr(i) != "'items'" and repr(i) != "'properties'"
            ]
            error_location = "->".join(repr(index) for index in error_path)
            if error_location:
                error_location = f"The error is in {error_location}."

            raise PipelineConfigError(
                f"Validation failed. {validation.message}. {error_location} " "See the stacktrace for more information."
            ) from validation

    logging.debug(f"Pipeline configuration is valid.")


def _overwrite_with_env_variables(component_definition: Dict[str, Any]):
    """
    Overwrite the pipeline config with environment variables. For example, to change index name param for an
    ElasticsearchDocumentStore, an env variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
    `_` sign must be used to specify nested hierarchical properties.

    :param definition: a dictionary containing the YAML definition of a component.
    """
    env_prefix = f"{component_definition['name']}_params_".upper()
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            param_name = key.replace(env_prefix, "").lower()
            component_definition["params"][param_name] = value
            logger.info(
                f"Param '{param_name}' of component '{component_definition['name']}' overwritten with environment variable '{key}' value '{value}'."
            )
