# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Any
import logging

import networkx  # type:ignore

from haystack.core.errors import PipelineValidationError
from haystack.core.component.sockets import InputSocket
from haystack.core.pipeline.descriptions import find_pipeline_inputs, describe_pipeline_inputs_as_string


logger = logging.getLogger(__name__)


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
        all_inputs = describe_pipeline_inputs_as_string(graph)
        raise ValueError(
            f"Pipeline received data for unknown component(s): {', '.join(unknown_components)}\n\n{all_inputs}"
        )

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
            inputs_for_node = input_values.get(node, {})
            missing_input_value = (
                inputs_for_node is None
                or not socket.name in inputs_for_node.keys()
                or inputs_for_node.get(socket.name, None) is None
            )
            if missing_input_value and socket.is_mandatory and not socket.is_variadic:
                all_inputs = describe_pipeline_inputs_as_string(graph)
                raise ValueError(f"Missing input: {node}.{socket.name}\n\n{all_inputs}")


def _validate_nodes_receive_only_expected_input(graph: networkx.MultiDiGraph, input_values: Dict[str, Any]):
    """
    Make sure that every input node is only receiving input values from EITHER the pipeline's input or another node,
    but never from both.
    """
    for node, input_data in input_values.items():
        for socket_name in input_data.keys():
            if input_data.get(socket_name, None) is None:
                continue
            if not socket_name in graph.nodes[node]["input_sockets"].keys():
                all_inputs = describe_pipeline_inputs_as_string(graph)
                raise ValueError(
                    f"Component {node} is not expecting any input value called {socket_name}.\n\n{all_inputs}"
                )

            input_socket: InputSocket = graph.nodes[node]["input_sockets"][socket_name]
            if input_socket.senders and not input_socket.is_variadic:
                raise ValueError(f"The input {socket_name} of {node} is already sent by: {input_socket.senders}")
