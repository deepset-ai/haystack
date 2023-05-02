from typing import Tuple, Optional, List, Type, TypeVar

import logging
import inspect
import itertools
from dataclasses import dataclass, fields

from pytypes import is_subtype as _is_subtype

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
    taken_by: Optional[str] = None
    variadic: bool = False


Socket = TypeVar("Socket", OutputSocket, InputSocket)


def is_subtype(from_what: Type, to_what: Type) -> bool:
    """
    Checks if two types are compatible
    """
    # TODO we should re-implement this method from pytypes if possible - pytypes is seriously unmaintained.
    return _is_subtype(from_what, to_what)

    # https://github.com/python/typing/issues/570
    # return from_what == to_what


def parse_connection_name(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def find_sockets(component):
    """
    Find a component's input and output sockets.
    """
    run_signature = inspect.signature(component.run)

    input_sockets = {}
    for param in run_signature.parameters:
        variadic = run_signature.parameters[param].kind == inspect.Parameter.VAR_POSITIONAL
        annotation = run_signature.parameters[param].annotation
        socket = InputSocket(name=run_signature.parameters[param].name, type=annotation, variadic=variadic)
        input_sockets[socket.name] = socket

    return_annotation = run_signature.return_annotation
    if return_annotation == inspect.Parameter.empty:
        return_annotation = component.output_type
    output_sockets = {field.name: OutputSocket(name=field.name, type=field.type) for field in fields(return_annotation)}

    return input_sockets, output_sockets


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
    possible_connections = [
        (out_sock, in_sock)
        for out_sock, in_sock in itertools.product(from_sockets, to_sockets)
        if not in_sock.taken_by and is_subtype(out_sock.type, in_sock.type)
    ]

    if not possible_connections:
        connections_status_str = connections_status(
            from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
        )
        raise PipelineConnectError(
            f"Cannot connect '{from_node}' with '{to_node}': "
            f"no matching connections available.\n{connections_status_str}"
        )

    if len(possible_connections) > 1:
        # Try to match by name
        name_matches = [
            (out_sock, in_sock) for out_sock, in_sock in possible_connections if in_sock.name == out_sock.name
        ]
        if len(name_matches) != 1:
            # TODO allow for multiple connections at once if there is no ambiguity?
            connections_status_str = connections_status(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )
            raise PipelineConnectError(
                f"Cannot connect '{from_node}' with '{to_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like "
                f"pipeline.connect(component_1.output_value, component_2.input_value).\n{connections_status_str}"
            )

    return possible_connections[0]


def locate_pipeline_input_components(graph) -> List[str]:
    """
    Collect the components with no input connections: they receive directly the pipeline inputs.
    """
    return [node for node in graph.nodes if not graph.in_edges(node)]


def locate_pipeline_output_components(graph) -> List[str]:
    """
    Collect the components with no output connections: these define the output of the pipeline.
    """
    return [node for node in graph.nodes if not graph.out_edges(node)]


def validate_pipeline(graph):
    """
    Make sure the pipeline has at least one input component and one output component.
    """
    if not locate_pipeline_input_components(graph):
        raise PipelineValidationError("This pipeline has no input components.")

    if not locate_pipeline_output_components(graph):
        raise PipelineValidationError("This pipeline has no output components.")
