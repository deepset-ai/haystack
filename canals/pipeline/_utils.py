from typing import Dict, Any, Callable, List, Union, Set

import sys
import json
import logging
from importlib import import_module
from inspect import getmembers, isclass

import networkx as nx


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


class PipelineSerializationError(PipelineError):
    pass


class PipelineDeserializationError(PipelineError):
    pass


class PipelineMaxLoops(PipelineError):
    pass


def find_nodes(modules_to_search: List[str]) -> Dict[str, Callable[..., Any]]:
    """
    Finds all functions decorated with `node` in all the modules listed in `modules_to_search`.

    WARNING: will attempt to import any module listed for search.

    Returns a dictionary with the node name and the node itself.
    """
    nodes: Dict[str, Any] = {}
    duplicate_names = []
    for search_module in modules_to_search:
        logger.debug("Searching for nodes under %s...", search_module)

        if not search_module in sys.modules.keys():
            logger.info("Importing %s to search for nodes...")
            import_module(search_module)

        for _, entity in getmembers(sys.modules[search_module], isclass):
            if hasattr(entity, "__canals_node__"):
                # It's a node
                if entity.__canals_node__ in nodes:
                    # Two nodes were discovered with the same name - namespace them
                    other_entity = nodes[entity.__canals_node__]
                    other_source_module = other_entity.__module__
                    logger.info(
                        "An node with the same name was found in two separate modules!\n"
                        " - Node name: %s\n - Found in modules: '%s' and '%s'\n"
                        "They both are going to be loaded, but you will need to use a namespace "
                        "path (%s.%s and %s.%s respectively) to use them in your Pipeline YAML definitions.",
                        entity.__canals_node__,
                        other_source_module,
                        search_module,
                        other_source_module,
                        entity.__canals_node__,
                        search_module,
                        entity.__canals_node__,
                    )
                    # Add both nodes as namespaced
                    nodes[f"{other_source_module}.{entity.__canals_node__}"] = other_entity
                    nodes[f"{search_module}.{entity.__canals_node__}"] = entity
                    # Do not remove the non-namespaced one, so in the case of a third collision it gets detected properly
                    duplicate_names.append(entity.__canals_node__)

                nodes[entity.__canals_node__] = entity
                logger.debug(" * Found node: %s", entity)

    # Now delete all remaining duplicates
    for duplicate in duplicate_names:
        del nodes[duplicate]

    print(nodes)
    return nodes


#
# FIXME REVIEW
#
def validate_graph(
    graph: nx.DiGraph,
    available_nodes: Dict[str, Dict[str, Union[str, Callable[..., Any]]]],
) -> None:
    """
    Makes sure the pipeline can run. Useful especially for pipelines loaded from file.
    """
    # Check that there are no isolated nodes or groups of nodes
    if not nx.is_weakly_connected(graph):
        raise PipelineValidationError(
            "The graph is not fully connected. Make sure all the nodes are connected to the same graph. "
            "You can use 'Pipeline.draw()' to visualize the graph, or inspect the 'Pipeline.graph' object."
        )

    # Check that the graph has starting nodes (nodes that take no input edges)
    input_nodes = [node for node in graph.nodes if not any(edge[1] == node for edge in graph.edges.data())]
    if not input_nodes:
        raise PipelineValidationError(
            "This pipeline doesn't seem to have starting nodes. "
            "Starting nodes are all nodes that have no input edges, "
            "plus all the nodes that were added to the Pipeline with "
            "input=True."
        )

    for node in graph.nodes:
        node = graph.nodes[node]["instance"]

        # Check that all nodes in the graph are actually registered nodes
        if not type(node) in available_nodes.values():
            raise PipelineValidationError(f"Node '{node}' not found. Are you sure it is a node?")

    logger.debug("Pipeline is valid")


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


def load_nodes(graph: nx.DiGraph, available_nodes: Dict[str, Dict[str, Callable[..., Any]]]) -> None:
    pass


#     """
#     Prepares the pipeline for the first execution. Instantiates all
#     class nodes present in the pipeline, if they're not instantiated yet.
#     """
#     # Convert node names into nodes and deserialize parameters
#     for name in graph.nodes:
#         try:
#             if isinstance(graph.nodes[name]["node"], str):
#                 graph.nodes[name]["node"] = available_nodes[
#                     graph.nodes[name]["node"]
#                 ]
#                 # If it's a class, check if it's reusable or needs instantiation
#                 if isclass(graph.nodes[name]["node"]):
#                     if "instance_id" in graph.nodes[name].keys():
#                         # Reusable: fish it out from the graph
#                         graph.nodes[name]["node"] = graph.nodes[
#                             graph.nodes[name]["instance_id"]
#                         ]["node"]
#                     else:
#                         # New: instantiate it
#                         graph.nodes[name]["node"] = graph.nodes[name]["node"](
#                             **graph.nodes[name]["init"] or {}
#                         )
#         except Exception as e:
#             raise PipelineDeserializationError(
#                 "Couldn't deserialize this node: " + name
#             ) from e

#         try:
#             if isinstance(graph.nodes[name]["parameters"], str):
#                 graph.nodes[name]["parameters"] = json.loads(
#                     graph.nodes[name]["parameters"]
#                 )
#         except Exception as e:
#             raise PipelineDeserializationError(
#                 "Couldn't deserialize this node's parameters: " + name
#             ) from e


def serialize(graph: nx.DiGraph) -> None:
    """
    Serializes all the nodes into a state that can be dumped to JSON or YAML.
    """
    reused_instances: Dict[str, Any] = {}
    for name in graph.nodes:
        # If the node is a reused instance, let's add the instance ID to the meta
        if graph.nodes[name]["instance"] in reused_instances.values():
            graph.nodes[name]["instance_id"] = [
                key for key, value in reused_instances.items() if value == graph.nodes[name]["instance"]
            ][0]

        elif hasattr(graph.nodes[name]["instance"], "init_parameters"):
            # Class nodes need to have a self.init_parameters attribute (or property)
            # if they want their init params to be serialized.
            try:
                graph.nodes[name]["init"] = graph.nodes[name]["instance"].init_parameters
            except Exception as e:
                raise PipelineSerializationError(
                    f"A node failed to provide its init parameters: {name}\n"
                    "If this is a custom node you wrote, you should save your init parameters into an instance "
                    "attribute called 'self.init_parameters' for this check to pass. "
                    "Add this step into your' '__init__' method."
                ) from e

            # This is a new node instance, so let's store it
            reused_instances[name] = graph.nodes[name]["instance"]

        # Serialize the callable by name
        try:
            graph.nodes[name]["instance"] = graph.nodes[name]["instance"].__canals_node__
        except Exception as e:
            raise PipelineSerializationError(f"Couldn't serialize this node: {name}")

        # Serialize its default parameters with JSON
        try:
            if graph.nodes[name]["parameters"]:
                graph.nodes[name]["parameters"] = json.dumps(graph.nodes[name]["parameters"])
        except Exception as e:
            raise PipelineSerializationError(f"Couldn't serialize this node's parameters: {name}")
