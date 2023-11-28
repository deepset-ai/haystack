# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Dict
import logging

import networkx  # type:ignore

from haystack.core.type_utils import _type_name
from haystack.core.component.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def find_pipeline_inputs(graph: networkx.MultiDiGraph) -> Dict[str, List[InputSocket]]:
    """
    Collect components that have disconnected input sockets. Note that this method returns *ALL* disconnected
    input sockets, including all such sockets with default values.
    """
    return {
        name: [socket for socket in data.get("input_sockets", {}).values() if not socket.senders or socket.is_variadic]
        for name, data in graph.nodes(data=True)
    }


def find_pipeline_outputs(graph: networkx.MultiDiGraph) -> Dict[str, List[OutputSocket]]:
    """
    Collect components that have disconnected output sockets. They define the pipeline output.
    """
    return {
        name: [socket for socket in data.get("output_sockets", {}).values() if not socket.receivers]
        for name, data in graph.nodes(data=True)
    }


def describe_pipeline_inputs(graph: networkx.MultiDiGraph):
    """
    Returns a dictionary with the input names and types that this pipeline accepts.
    """
    inputs = {
        comp: {socket.name: {"type": socket.type, "is_mandatory": socket.is_mandatory} for socket in data}
        for comp, data in find_pipeline_inputs(graph).items()
        if data
    }
    return inputs


def describe_pipeline_inputs_as_string(graph: networkx.MultiDiGraph):
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
