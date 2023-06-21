# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Dict, Any
import logging
from dataclasses import fields

import networkx

from canals.errors import PipelineValidationError
from canals.pipeline.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def find_pipeline_inputs(graph: networkx.MultiDiGraph) -> Dict[str, List[InputSocket]]:
    """
    Collect components that have disconnected input sockets. Note that this method returns *ALL* disconnected
    input sockets, including all such sockets with default values.
    """
    return {
        node: [socket for socket in data.get("input_sockets", {}).values() if not socket.sender]
        for node, data in graph.nodes(data=True)
    }


def find_pipeline_outputs(graph) -> Dict[str, List[OutputSocket]]:
    """
    Collect components that have disconnected output sockets. They define the pipeline output.
    """
    return {
        node: list(data.get("output_sockets", {}).values())
        for node, data in graph.nodes(data=True)
        if not graph.out_edges(node)
    }


def validate_pipeline_input(graph: networkx.MultiDiGraph, input_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure the pipeline is properly built and that the input received makes sense.
    Returns the input values, validated and updated at need.
    """
    if not any(sockets for sockets in find_pipeline_inputs(graph).values()):
        raise PipelineValidationError("This pipeline has no inputs.")

    # Make sure the input keys are all nodes of the pipeline
    unknown_components = [key for key in input_values.keys() if not key in graph.nodes]
    if unknown_components:
        raise ValueError(f"Pipeline received data for unknown component(s): {', '.join(unknown_components)}")

    # Make sure all necessary sockets are connected
    _validate_input_sockets_are_connected(graph, input_values)

    # Make sure that the pipeline input is only sent to nodes that won't receive data from other nodes
    _validate_nodes_receive_only_expected_input(graph, input_values)

    return input_values


def _validate_input_sockets_are_connected(graph: networkx.MultiDiGraph, input_values: Dict[str, Any]):
    """
    Make sure all the inputs nodes are receiving all the values they need, either from the Pipeline's input or from
    other nodes.
    """
    valid_inputs = find_pipeline_inputs(graph)
    for node, sockets in valid_inputs.items():
        for socket in sockets:
            node_instance = graph.nodes[node]["instance"]
            input_in_node_defaults = hasattr(node_instance, "defaults") and socket.name in node_instance.defaults
            inputs_for_node = input_values.get(node)
            missing_input_value = (
                not inputs_for_node
                or not socket.name in [f.name for f in fields(inputs_for_node)]
                or not getattr(inputs_for_node, socket.name)
            )
            if missing_input_value and not input_in_node_defaults:
                raise ValueError(f"Missing input: {node}.{socket.name}")


def _validate_nodes_receive_only_expected_input(graph: networkx.MultiDiGraph, input_values: Dict[str, Any]):
    """
    Make sure that every input node is only receiving input values from EITHER the pipeline's input or another node,
    but never from both.
    """
    for node, input_data in input_values.items():
        for socket_name in [f.name for f in fields(input_data)]:
            if not getattr(input_data, socket_name):
                continue
            if not socket_name in graph.nodes[node]["input_sockets"].keys():
                raise ValueError(
                    f"Component {node} is not expecting any input value called {socket_name}. "
                    "Are you using the correct Input class?"
                )

            sender = graph.nodes[node]["input_sockets"][socket_name].sender
            if sender:
                raise ValueError(f"The input {socket_name} of {node} is already sent by node {sender}")
