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
    inputs: dict[str, dict[str, InputSocketTypeDescriptor]] = {
        comp: {socket.name: {"type": socket.type, "is_mandatory": socket.is_mandatory} for socket in data}
        for comp, data in find_pipeline_inputs(graph).items()
        if data
    }
    return inputs


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
