import inspect
import logging
import re
import sys
import networkx as nx
from typing import Any, Dict, List, Optional
from networkx import DiGraph
from haystack.pipelines.config import (
    build_component_dependency_graph,
    get_component_definitions,
    get_pipeline_definition,
    validate_config,
)

from haystack.schema import EvaluationResult


logger = logging.getLogger(__name__)


MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
CODE_GEN_ALLOWED_IMPORTS = ["haystack.document_stores", "haystack.nodes", "haystack.pipelines"]
CAMEL_CASE_TO_SNAKE_CASE_REGEX = re.compile(r"(?<=[a-z])(?=[A-Z0-9])")


def camel_to_snake_case(input: str) -> str:
    return CAMEL_CASE_TO_SNAKE_CASE_REGEX.sub("_", input).lower()


def generate_code(
    pipeline_config: Dict[str, Any],
    pipeline_variable_name: str = "pipeline",
    pipeline_name: Optional[str] = None,
    generate_imports: bool = True,
    comment: Optional[str] = None,
    add_pipeline_cls_import: bool = True,
) -> str:
    """
    Generates code to create a pipeline.

    :param pipeline_config: The pipeline config that specifies components and pipelines.
    :param pipeline_variable_name: The variable name of the pipeline to be generated.
                                   Defaults to "pipeline".
    :param pipeline_name: The name of the pipeline defined in pipeline_config to be generated.
                          If unspecified the first pipeline will be taken.
    :param generate_imports: Whether to generate imports code.
                             Defaults to True.
    :param comment: Preceding Comment to add to the code.
    :param add_pipeline_cls_import: Whether to add import statement for Pipeline class if generate_imports is True.
                                    Defaults to True.
    """
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
        imports_code = _generate_imports_code(types_to_import=types_to_import)
        code_parts.append(imports_code)

    components_code = _generate_components_code(
        component_definitions=component_definitions,
        component_variable_names=component_variable_names,
        dependency_graph=component_dependency_graph,
    )
    pipeline_code = _generate_pipeline_code(
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


def _generate_pipeline_code(
    pipeline_definition: Dict[str, Any], component_variable_names: Dict[str, str], pipeline_variable_name: str
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


def _generate_components_code(
    component_definitions: Dict[str, Any], component_variable_names: Dict[str, str], dependency_graph: DiGraph
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


def _generate_imports_code(types_to_import: List[str]) -> str:
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


def print_eval_report(
    eval_result: EvaluationResult,
    graph: DiGraph,
    n_wrong_examples: int = 3,
    metrics_filter: Optional[Dict[str, List[str]]] = None,
):
    """
    Prints a report for a given EvaluationResult visualizing metrics per node specified by the pipeline graph.

    :param eval_result: The evaluation result.
    :param graph: The graph of the pipeline containing the node names and its connections.
    :param n_wrong_examples: The number of examples to show in order to inspect wrong predictions.
                             Defaults to 3.
    :param metrics_filter: Specifies which metrics of eval_result to show in the report.
    """
    if any(degree > 1 for node, degree in graph.out_degree):
        logger.warning("Pipelines with junctions are currently not supported.")
        return

    calculated_metrics = {
        "": eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match"),
        "_top_1": eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match", simulated_top_k_reader=1),
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

    pipeline_overview = _format_pipeline_overview(calculated_metrics=calculated_metrics, graph=graph)
    wrong_examples_report = _format_wrong_examples_report(eval_result=eval_result, n_wrong_examples=n_wrong_examples)

    print(f"{pipeline_overview}\n" f"{wrong_examples_report}")


def _format_document_answer(document_or_answer: dict):
    return "\n \t".join(f"{name}: {value}" for name, value in document_or_answer.items())


def _format_wrong_example(query: dict):
    metrics = "\n \t".join(f"{name}: {value}" for name, value in query["metrics"].items())
    documents = "\n\n \t".join(map(_format_document_answer, query.get("documents", [])))
    documents = f"Documents: \n \t{documents}\n" if len(documents) > 0 else ""
    answers = "\n\n \t".join(map(_format_document_answer, query.get("answers", [])))
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


def _format_wrong_examples_node(node_name: str, wrong_examples_formatted: str):
    s = (
        f"                Wrong {node_name} Examples\n"
        f"=======================================================\n"
        f"{wrong_examples_formatted}\n"
        f"=======================================================\n"
    )
    return s


def _format_wrong_examples_report(eval_result: EvaluationResult, n_wrong_examples: int = 3):
    examples = {
        node: eval_result.wrong_examples(node, doc_relevance_col="gold_id_or_answer_match", n=n_wrong_examples)
        for node in eval_result.node_results.keys()
    }
    examples_formatted = {node: "\n".join(map(_format_wrong_example, examples)) for node, examples in examples.items()}

    return "\n".join(map(_format_wrong_examples_node, examples_formatted.keys(), examples_formatted.values()))


def _format_pipeline_node(node: str, calculated_metrics: dict):
    node_metrics: dict = {}
    for metric_mode, metrics in calculated_metrics.items():
        for metric, value in metrics.get(node, {}).items():
            node_metrics[f"{metric}{metric_mode}"] = value

    def format_node_metric(metric, value):
        return f"                        | {metric}: {value:5.3}"

    node_metrics_formatted = "\n".join(sorted(map(format_node_metric, node_metrics.keys(), node_metrics.values())))
    node_metrics_formatted = f"{node_metrics_formatted}\n" if len(node_metrics_formatted) > 0 else ""
    s = (
        f"                      {node}\n"
        f"                        |\n"
        f"{node_metrics_formatted}"
        f"                        |"
    )
    return s


def _format_pipeline_overview(calculated_metrics: dict, graph: DiGraph):
    pipeline_overview = "\n".join(_format_pipeline_node(node, calculated_metrics) for node in graph.nodes)
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
