import copy
import inspect
import logging
import os
from pathlib import Path
import re
import sys
import networkx as nx
from typing import Any, Dict, List, Optional
from networkx import DiGraph
import yaml

from haystack.schema import EvaluationResult


logger = logging.getLogger(__name__)


CAMEL_CASE_TO_SNAKE_CASE_REGEX = re.compile(r"(?<=[a-z])(?=[A-Z0-9])")
VALID_CODE_GEN_INPUT_REGEX = re.compile(r"^[-a-zA-Z0-9_/.:]+$")
MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
CODE_GEN_ALLOWED_IMPORTS = ["haystack.document_stores", "haystack.nodes", "haystack.pipelines"]


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
            raise Exception("The YAML contains multiple pipelines. Please specify the pipeline name to load.")
    else:
        pipelines_in_definitions = list(filter(lambda p: p["name"] == pipeline_name, pipeline_config["pipelines"]))
        if not pipelines_in_definitions:
            raise KeyError(f"Cannot find any pipeline with name '{pipeline_name}' declared in the YAML file.")
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


def read_pipeline_config_from_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_config(pipeline_config: Dict[str, Any]):
    for component in pipeline_config["components"]:
        _validate_user_input(component["name"])
        _validate_user_input(component["type"])
        for k, v in component.get("params", {}).items():
            _validate_user_input(k)
            _validate_user_input(v)
    for pipeline in pipeline_config["pipelines"]:
        _validate_user_input(pipeline["name"])
        _validate_user_input(pipeline["type"])
        for node in pipeline["nodes"]:
            _validate_user_input(node["name"])
            for input in node["inputs"]:
                _validate_user_input(input)


def camel_to_snake_case(input: str) -> str:
    return CAMEL_CASE_TO_SNAKE_CASE_REGEX.sub("_", input).lower()


def build_component_dependency_graph(
    pipeline_definition: Dict[str, Any], component_definitions: Dict[str, Any]
) -> DiGraph:
    graph = DiGraph()
    for node in pipeline_definition["nodes"]:
        node_name = node["name"]
        graph.add_node(node_name)
        for input in node["inputs"]:
            if input in component_definitions:
                graph.add_edge(input, node_name)
    for component_name, component_definition in component_definitions.items():
        params = component_definition.get("params", {})
        referenced_components: List[str] = list()
        for param_value in params.values():
            if param_value in component_definitions:
                referenced_components.append(param_value)
        for referenced_component in referenced_components:
            graph.add_edge(referenced_component, component_name)
    return graph


def _validate_user_input(input: str):
    if isinstance(input, str) and not VALID_CODE_GEN_INPUT_REGEX.match(input):
        raise ValueError(f"'{input}' is not a valid code gen variable name. Use word characters only.")


def _overwrite_with_env_variables(definition: Dict[str, Any]):
    """
    Overwrite the pipeline config with environment variables. For example, to change index name param for an
    ElasticsearchDocumentStore, an env variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
    `_` sign must be used to specify nested hierarchical properties.

    :param definition: a dictionary containing the YAML definition of a component.
    """
    env_prefix = f"{definition['name']}_params_".upper()
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            param_name = key.replace(env_prefix, "").lower()
            definition["params"][param_name] = value


class _PipelineCodeGen:
    @classmethod
    def generate_code(
        cls,
        pipeline_config: Dict[str, Any],
        pipeline_variable_name: str = "pipeline",
        pipeline_name: Optional[str] = None,
        generate_imports: bool = True,
        comment: Optional[str] = None,
        add_pipeline_cls_import: bool = True,
    ) -> str:
        validate_config(pipeline_config)

        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config, overwrite_with_env_variables=False
        )
        component_variable_names = {name: camel_to_snake_case(name) for name in component_definitions.keys()}
        pipeline_definition = get_pipeline_definition(pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_dependency_graph = build_component_dependency_graph(
            pipeline_definition=pipeline_definition, component_definitions=component_definitions
        )

        code_parts = []
        if generate_imports:
            types_to_import = [component["type"] for component in component_definitions.values()]
            if add_pipeline_cls_import:
                types_to_import.append("Pipeline")
            imports_code = cls._generate_imports_code(types_to_import=types_to_import)
            code_parts.append(imports_code)

        components_code = cls._generate_components_code(
            component_definitions=component_definitions,
            component_variable_names=component_variable_names,
            dependency_graph=component_dependency_graph,
        )
        pipeline_code = cls._generate_pipeline_code(
            pipeline_definition=pipeline_definition,
            component_variable_names=component_variable_names,
            pipeline_variable_name=pipeline_variable_name,
        )

        code_parts.append(components_code)
        code_parts.append(pipeline_code)
        code = "\n\n".join(code_parts)

        if comment:
            comment = re.sub(r"^(#\s)?", "# ", comment, flags=re.MULTILINE)
            code = "\n".join([comment, code])

        return code

    @classmethod
    def _generate_pipeline_code(
        cls, pipeline_definition: Dict[str, Any], component_variable_names: Dict[str, str], pipeline_variable_name: str
    ) -> str:
        code_lines = [f"{pipeline_variable_name} = Pipeline()"]
        for node in pipeline_definition["nodes"]:
            node_name = node["name"]
            component_variable_name = component_variable_names[node_name]
            inputs = ", ".join(f'"{name}"' for name in node["inputs"])
            code_lines.append(
                f'{pipeline_variable_name}.add_node(component={component_variable_name}, name="{node_name}", inputs=[{inputs}])'
            )

        code = "\n".join(code_lines)
        return code

    @classmethod
    def _generate_components_code(
        cls, component_definitions: Dict[str, Any], component_variable_names: Dict[str, str], dependency_graph: DiGraph
    ) -> str:
        code = ""
        declarations = {}
        for name, definition in component_definitions.items():
            variable_name = component_variable_names[name]
            class_name = definition["type"]
            param_value_dict = {
                key: component_variable_names.get(value, f'"{value}"') if type(value) == str else value
                for key, value in definition.get("params", {}).items()
            }
            init_args = ", ".join(f"{key}={value}" for key, value in param_value_dict.items())
            declarations[name] = f"{variable_name} = {class_name}({init_args})"

        ordered_components = nx.topological_sort(dependency_graph)
        ordered_declarations = [declarations[component] for component in ordered_components]
        code = "\n".join(ordered_declarations)
        return code

    @classmethod
    def _generate_imports_code(cls, types_to_import: List[str]) -> str:
        code_lines = []
        importable_classes = {
            name: mod
            for mod in CODE_GEN_ALLOWED_IMPORTS
            for name, obj in inspect.getmembers(sys.modules[mod])
            if inspect.isclass(obj)
        }

        imports_by_module: Dict[str, List[str]] = {}
        for t in types_to_import:
            mod = importable_classes.get(t, MODULE_NOT_FOUND)
            if mod in imports_by_module:
                imports_by_module[mod].append(t)
            else:
                imports_by_module[mod] = [t]

        for mod in sorted(imports_by_module.keys()):
            sorted_types = sorted(set(imports_by_module[mod]))
            import_types = ", ".join(sorted_types)
            line_prefix = "# " if mod == MODULE_NOT_FOUND else ""
            code_lines.append(f"{line_prefix}from {mod} import {import_types}")

        code = "\n".join(code_lines)
        return code


class _PipelineEvalReportGen:
    @classmethod
    def print_eval_report(
        cls,
        eval_result: EvaluationResult,
        graph: DiGraph,
        n_wrong_examples: int = 3,
        metrics_filter: Optional[Dict[str, List[str]]] = None,
    ):
        if any(degree > 1 for node, degree in graph.out_degree):
            logger.warning("Pipelines with junctions are currently not supported.")
            return

        calculated_metrics = {
            "": eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match"),
            "_top_1": eval_result.calculate_metrics(
                doc_relevance_col="gold_id_or_answer_match", simulated_top_k_reader=1
            ),
            " upper bound": eval_result.calculate_metrics(
                doc_relevance_col="gold_id_or_answer_match", eval_mode="isolated"
            ),
        }

        if metrics_filter is not None:
            for metric_mode in calculated_metrics:
                calculated_metrics[metric_mode] = {
                    node: metrics
                    if node not in metrics_filter
                    else {metric: value for metric, value in metrics.items() if metric in metrics_filter[node]}
                    for node, metrics in calculated_metrics[metric_mode].items()
                }

        pipeline_overview = cls._format_pipeline_overview(calculated_metrics=calculated_metrics, graph=graph)
        wrong_samples_report = cls._format_wrong_samples_report(
            eval_result=eval_result, n_wrong_examples=n_wrong_examples
        )

        print(f"{pipeline_overview}\n" f"{wrong_samples_report}")

    @classmethod
    def _format_document_answer(cls, document_or_answer: dict):
        return "\n \t".join([f"{name}: {value}" for name, value in document_or_answer.items()])

    @classmethod
    def _format_wrong_sample(cls, query: dict):
        metrics = "\n \t".join([f"{name}: {value}" for name, value in query["metrics"].items()])
        documents = "\n\n \t".join([cls._format_document_answer(doc) for doc in query.get("documents", [])])
        documents = f"Documents: \n \t{documents}\n" if len(documents) > 0 else ""
        answers = "\n\n \t".join([cls._format_document_answer(answer) for answer in query.get("answers", [])])
        answers = f"Answers: \n \t{answers}\n" if len(answers) > 0 else ""
        gold_document_ids = "\n \t".join(query["gold_document_ids"])
        gold_answers = "\n \t".join(query.get("gold_answers", []))
        gold_answers = f"Gold Answers: \n \t{gold_answers}\n" if len(gold_answers) > 0 else ""
        s = (
            f"Query: \n \t{query['query']}\n"
            f"{gold_answers}"
            f"Gold Document Ids: \n \t{gold_document_ids}\n"
            f"Metrics: \n \t{metrics}\n"
            f"{answers}"
            f"{documents}"
            f"_______________________________________________________"
        )
        return s

    @classmethod
    def _format_wrong_samples_node(cls, node_name: str, wrong_samples_formatted: str):
        s = (
            f"                Wrong {node_name} Examples\n"
            f"=======================================================\n"
            f"{wrong_samples_formatted}\n"
            f"=======================================================\n"
        )
        return s

    @classmethod
    def _format_wrong_samples_report(cls, eval_result: EvaluationResult, n_wrong_examples: int = 3):
        examples = {
            node: eval_result.wrong_examples(node, doc_relevance_col="gold_id_or_answer_match", n=n_wrong_examples)
            for node in eval_result.node_results.keys()
        }
        examples_formatted = {
            node: "\n".join([cls._format_wrong_sample(example) for example in examples])
            for node, examples in examples.items()
        }

        return "\n".join(
            [cls._format_wrong_samples_node(node, examples) for node, examples in examples_formatted.items()]
        )

    @classmethod
    def _format_pipeline_node(cls, node: str, calculated_metrics: dict):
        node_metrics: dict = {}
        for metric_mode in calculated_metrics:
            for metric, value in calculated_metrics[metric_mode].get(node, {}).items():
                node_metrics[f"{metric}{metric_mode}"] = value

        node_metrics_formatted = "\n".join(
            sorted([f"                        | {metric}: {value:5.3}" for metric, value in node_metrics.items()])
        )
        node_metrics_formatted = f"{node_metrics_formatted}\n" if len(node_metrics_formatted) > 0 else ""
        s = (
            f"                      {node}\n"
            f"                        |\n"
            f"{node_metrics_formatted}"
            f"                        |"
        )
        return s

    @classmethod
    def _format_pipeline_overview(cls, calculated_metrics: dict, graph: DiGraph):
        pipeline_overview = "\n".join([cls._format_pipeline_node(node, calculated_metrics) for node in graph.nodes])
        s = (
            f"================== Evaluation Report ==================\n"
            f"=======================================================\n"
            f"                   Pipeline Overview\n"
            f"=======================================================\n"
            f"{pipeline_overview}\n"
            f"                      Output\n"
            f"=======================================================\n"
        )
        return s
