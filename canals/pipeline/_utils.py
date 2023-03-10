from typing import Dict, TYPE_CHECKING, List, Any, Tuple

import json
import logging
from pathlib import Path

if TYPE_CHECKING:
    from canals import Pipeline


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineMaxLoops(PipelineError):
    pass


def locate_pipeline_input_nodes(graph) -> List[str]:
    """
    Collect the nodes with no input edges: they receive directly the pipeline inputs.
    """
    return [node for node in graph.nodes if not graph.in_edges(node) or graph.nodes[node]["input_node"]]


def locate_pipeline_output_nodes(graph) -> List[str]:
    """
    Collect the nodes with no output edges: these define the output of the pipeline.
    """
    return [node for node in graph.nodes if not graph.out_edges(node) or graph.nodes[node]["output_node"]]


def _discover_dependencies(nodes: List[object]) -> List[str]:
    """
    Given a list of nodes, it returns a list of all the modules that one needs to import to
    make this pipeline work.
    """
    return list({node.__module__.split(".")[0] for node in nodes}) + ["canals"]


def save_pipelines(
    pipelines: Dict[str, "Pipeline"], path: Path, _writer=lambda obj, handle: json.dump(obj, handle, indent=4)
) -> None:
    """
    Converts a dictionary of named Pipelines into a JSON file.

    :param pipelines: dictionary of {name: pipeline_object}
    :param path: where to write the resulting file
    :param _writer: which function to use to write the dictionary to a file.
        Use this parameter to dump to a different format like YAML, TOML, HCL, etc.
    """
    result = marshal_pipelines(pipelines=pipelines)
    with open(path, "w", encoding="utf-8") as handle:
        _writer(result, handle)


def marshal_pipelines(pipelines: Dict[str, "Pipeline"]) -> Dict[str, Any]:
    """
    Converts a dictionary of named Pipelines into a Python dictionary that can be
    written to a JSON file.

    :param pipelines: dictionary of {name: pipeline_object}
    """
    result: Dict[str, Any] = {}

    # Summarize pipeline configuration
    nodes: List[Tuple[str, str, object]] = []
    result["pipelines"] = {}
    for pipeline_name, pipeline in pipelines.items():
        pipeline_repr: Dict[str, Any] = {}
        pipeline_repr["metadata"] = pipeline.metadata
        pipeline_repr["max_loops_allowed"] = pipeline.max_loops_allowed
        pipeline_repr["edges"] = list(pipeline.graph.edges)

        # Collect nodes
        pipeline_repr["nodes"] = {}
        for node_name in pipeline.graph.nodes:

            # Check if we saved the same instance twice (or more times) and replace duplicates with references.
            node_instance = pipeline.graph.nodes[node_name]["instance"]
            for existing_node_pipeline, existing_node_name, existing_node in nodes:
                if node_instance == existing_node:
                    # Build the pointer - done this way to support languages with no pointer syntax (e.g. JSON)
                    if existing_node_pipeline == pipeline_name:
                        pipeline_repr["nodes"][node_name] = {"refer_to": existing_node_name}
                    else:
                        pipeline_repr["nodes"][node_name] = {
                            "refer_to": f"{existing_node_pipeline}.{existing_node_name}"
                        }
                    break

            # If no pointer was made in the previous step
            if not node_name in pipeline_repr["nodes"]:
                # Save the node in the global nodes list
                nodes.append((pipeline_name, node_name, node_instance))
                # Serialize the node
                node_repr = {"type": node_instance.__class__.__name__, "init_parameters": node_instance.init_parameters}
                pipeline_repr["nodes"][node_name] = node_repr

            # Check for run parameters
            if pipeline.graph.nodes[node_name]["parameters"]:
                pipeline_repr["nodes"][node_name]["run_parameters"] = pipeline.graph.nodes[node_name]["parameters"]

        result["pipelines"][pipeline_name] = pipeline_repr

    # Collect the dependencies
    result["dependencies"] = _discover_dependencies(nodes=[node[2] for node in nodes])
    return result


def load_pipelines(path: Path, _reader=json.load) -> Dict[str, "Pipeline"]:
    """
    Reads the given file and returns a dictionary of named Pipelines ready to use.

    :param path: the path of the pipelines file.
    :param _reader: which function to use to read the dictionary from file.
        Use this parameter to read from a different format like YAML, TOML, HCL, etc.
    """
    return {}
