from typing import Tuple, Optional, List, Iterable, Dict, Any

import logging
import inspect
import itertools
from dataclasses import dataclass, fields

import networkx

from canals.errors import PipelineConnectError, PipelineValidationError


logger = logging.getLogger(__name__)


@dataclass
class OutputSocket:
    name: str
    type: type


@dataclass
class InputSocket:
    name: str
    type: type
    variadic: bool
    taken_by: Optional[str] = None


def parse_connection_name(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def find_input_sockets(component) -> Dict[str, InputSocket]:
    """
    Find a component's input sockets.
    """
    run_signature = inspect.signature(component.run)

    input_sockets = {}
    for param in run_signature.parameters:
        name = run_signature.parameters[param].name
        variadic = run_signature.parameters[param].kind == inspect.Parameter.VAR_POSITIONAL
        annotation = run_signature.parameters[param].annotation

        socket = InputSocket(name=name, type=annotation, variadic=variadic)
        input_sockets[socket.name] = socket

    return input_sockets


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    run_signature = inspect.signature(component.run)

    return_annotation = run_signature.return_annotation
    if return_annotation == inspect.Parameter.empty:
        return_annotation = component.output_type

    output_sockets = {field.name: OutputSocket(name=field.name, type=field.type) for field in fields(return_annotation)}

    return output_sockets


def connections_status(from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]):
    """
    Lists the status of the sockets, for error messages.
    """
    from_sockets_list = "\n".join([f" - {socket.name} ({socket.type.__name__})" for socket in from_sockets])
    to_sockets_list = "\n".join(
        [
            f" - {socket.name} ({socket.type.__name__}, {'taken by '+socket.taken_by if socket.taken_by else 'available'})"
            for socket in to_sockets
        ]
    )
    return f"'{from_node}':\n{from_sockets_list}\n'{to_node}':\n{to_sockets_list}"


def find_unambiguous_connection(
    from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]
) -> Tuple[OutputSocket, InputSocket]:
    """
    Find one single possible connection between two lists of sockets.
    """
    # List all combinations of sockets that match by type
    possible_connections = [
        (out_sock, in_sock)
        for out_sock, in_sock in itertools.product(from_sockets, to_sockets)
        if not in_sock.taken_by and out_sock.type == in_sock.type
    ]

    # No connections seem to be possible
    if not possible_connections:
        connections_status_str = connections_status(
            from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
        )
        raise PipelineConnectError(
            f"Cannot connect '{from_node}' with '{to_node}': "
            f"no matching connections available.\n{connections_status_str}"
        )

    # There's more than one possible connection
    if len(possible_connections) > 1:
        # Try to match by name
        name_matches = [
            (out_sock, in_sock) for out_sock, in_sock in possible_connections if in_sock.name == out_sock.name
        ]
        if len(name_matches) != 1:
            # TODO allow for multiple connections at once if there is no ambiguity?
            # TODO give priority to sockets that have no default values?
            connections_status_str = connections_status(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )
            raise PipelineConnectError(
                f"Cannot connect '{from_node}' with '{to_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like: "
                f"pipeline.connect('component_1.output_value', 'component_2.input_value').\n{connections_status_str}"
            )

    return possible_connections[0]


def find_pipeline_inputs(graph: networkx.MultiDiGraph) -> Dict[str, List[InputSocket]]:
    """
    Collect components that have disconnected input sockets. Note that this method returns *ALL* disconnected
    input sockets, including all such sockets with default values.
    """
    return {
        node: [socket for socket in data.get("input_sockets", {}).values() if not socket.taken_by]
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


def _validate_input_sockets_are_connected(graph: networkx.MultiDiGraph, inputs_values: Dict[str, Dict[str, Any]]):
    valid_inputs = find_pipeline_inputs(graph)
    for node, sockets in valid_inputs.items():
        if node in inputs_values:
            for socket in sockets:
                node_instance = graph.nodes[node]["instance"]
                input_in_node_defaults = hasattr(node_instance, "defaults") and socket.name in node_instance.defaults
                if not input_in_node_defaults and not socket.name in inputs_values[node]:
                    raise ValueError(f"Missing input: {node}.{socket.name}")


def _validate_nodes_receive_only_expected_input(graph: networkx.MultiDiGraph, inputs_values: Dict[str, Dict[str, Any]]):
    for node, input_data in inputs_values.items():
        for socket_name in input_data.keys():
            if not socket_name in graph.nodes[node]["input_sockets"].keys():
                raise ValueError(f"Component {node} is not expecting any input value called {socket_name}")

            taken_by = graph.nodes[node]["input_sockets"][socket_name].taken_by
            if taken_by:
                raise ValueError(f"The input {socket_name} of {node} is already taken by node {taken_by}")


def validate_pipeline_input(  # pylint: disable=too-many-branches
    graph: networkx.MultiDiGraph, inputs_values: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Make sure the pipeline is properly built and that the input received makes sense.

    Returns the input values, validated and updated at need.
    """
    input_components = find_pipeline_inputs(graph)
    if not find_pipeline_inputs(graph):
        raise PipelineValidationError("This pipeline has no inputs.")

    # Make sure the input keys are all nodes of the pipeline
    unknown_components = [key for key in inputs_values.keys() if not key in graph.nodes]
    if unknown_components:
        raise ValueError(f"Pipeline received data for unknown component(s): {', '.join(unknown_components)}")

    # Make sure all necessary sockets are connected
    _validate_input_sockets_are_connected(graph, inputs_values)

    # Make sure that the pipeline input is only sent to nodes that won't receive data from other nodes
    _validate_nodes_receive_only_expected_input(graph, inputs_values)

    # Make sure variadic input components are receiving lists
    for component in input_components.keys():
        if graph.nodes[component]["variadic_input"] and component in inputs_values.keys():
            for key, value in inputs_values[component].items():  # should be just one
                if not isinstance(value, Iterable):
                    inputs_values[component][key] = [value]

    return inputs_values
