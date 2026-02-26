# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import networkx

from haystack.core.component.types import InputSocket, InputSocketTypeDescriptor, OutputSocket
from haystack.core.type_utils import _type_name


def find_pipeline_inputs(
    graph: networkx.MultiDiGraph, include_connected_sockets: bool = False
) -> dict[str, list[InputSocket]]:
    """
    Collect components that have disconnected/connected input sockets.

    Note that this method returns *ALL* disconnected input sockets, including all such sockets with default values.
    It also includes variadic input sockets, even if they are currently connected, as they can accept additional
    inputs from outside the pipeline.

    :param graph: The pipeline graph to analyze.
    :param include_connected_sockets: If True, also include input sockets that are already connected.
        This can be useful for understanding the full input requirements of the pipeline, including inputs
        that are currently satisfied by connections within the pipeline. If False, only include input sockets that
        are not connected to any output socket, which represent the external inputs that can be provided when running
        the pipeline.
    """
    return {
        name: [
            socket
            for socket in data.get("input_sockets", {}).values()
            if socket.is_variadic or (include_connected_sockets or not socket.senders)
        ]
        for name, data in graph.nodes(data=True)
    }


def find_pipeline_outputs(
    graph: networkx.MultiDiGraph, include_connected_sockets: bool = False
) -> dict[str, list[OutputSocket]]:
    """
    Collect components that have disconnected/connected output sockets. They define the pipeline output.
    """
    return {
        name: [
            socket
            for socket in data.get("output_sockets", {}).values()
            if (include_connected_sockets or not socket.receivers)
        ]
        for name, data in graph.nodes(data=True)
    }


def describe_pipeline_inputs(graph: networkx.MultiDiGraph) -> dict[str, dict[str, InputSocketTypeDescriptor]]:
    """
    Returns a dictionary with the input names and types that this pipeline accepts.
    """
    return {
        comp: {
            socket.name: InputSocketTypeDescriptor(
                type=socket.type,
                # Variadic mandatory sockets with existing connections don't require user input, so treat them as
                # optional.
                is_mandatory=socket.is_mandatory and socket.senders == [],
            )
            for socket in sockets
        }
        for comp, sockets in find_pipeline_inputs(graph).items()
    }


def describe_pipeline_inputs_as_string(graph: networkx.MultiDiGraph) -> str:
    """
    Returns a string representation of the input names and types that this pipeline accepts.
    """
    inputs = describe_pipeline_inputs(graph)
    message = "This pipeline expects the following inputs:\n"
    for comp, sockets in inputs.items():
        if sockets:
            message += f"- {comp}:\n"
            for name, socket in sockets.items():
                message += f"    - {name}: {_type_name(socket['type'])}\n"
    return message
