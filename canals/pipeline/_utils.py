from typing import Dict

import json
import logging
from pathlib import Path

from canals import Pipeline


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class NoSuchStoreError(PipelineError):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineMaxLoops(PipelineError):
    pass


def locate_pipeline_input_nodes(graph):
    """
    Collect the nodes with no input edges: they receive directly the pipeline inputs.
    """
    return [node for node in graph.nodes if not graph.in_edges(node) or graph.nodes[node]["input_node"]]


def locate_pipeline_output_nodes(graph):
    """
    Collect the nodes with no output edges: these define the output of the pipeline.
    """
    return [node for node in graph.nodes if not graph.out_edges(node) or graph.nodes[node]["output_node"]]


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer=json.dump) -> None:
    """
    Converts a dictionary of named Pipelines into a Python dictionary that can be
    written to a JSON file.

    :param pipelines: dictionary of {name: pipeline_object}
    :param path: where to write the resulting file
    :param _writer: which function to use to write the dictionary to a file.
        Use this parameter to dump to a different format like YAML, TOML, HCL, etc.
    """
    pass


def load_pipelines(path: Path, _reader=json.load) -> Dict[str, Pipeline]:
    """
    Reads the given file and returns a dictionary of named Pipelines ready to use.

    :param path: the path of the pipelines file.
    :param _reader: which function to use to read the dictionary from file.
        Use this parameter to read from a different format like YAML, TOML, HCL, etc.
    """
    return {}
