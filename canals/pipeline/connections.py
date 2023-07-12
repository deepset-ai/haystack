# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, List, Any, get_args, get_origin

import logging
import itertools

from canals.errors import PipelineConnectError
from canals.pipeline.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def parse_connection_name(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def type_is_compatible(source_type, dest_type):
    """
    Checks whether the source type is equal or a subtype of the destination type.

    Used to validate pipeline connections.
    """
    if source_type == dest_type or dest_type is Any:
        return True

    if source_type is Any:
        return False

    try:
        print(source_type, dest_type)
        if issubclass(source_type, dest_type):
            return True
    except TypeError:
        return False

    source_origin = get_origin(source_type)
    dest_origin = get_origin(dest_type)
    if not source_origin or not dest_origin or source_origin != dest_origin:
        return False

    args_pairs = zip(get_args(source_type), get_args(dest_type))
    return all(type_is_compatible(*args) for args in args_pairs)


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
        if not in_sock.sender and all(type_is_compatible(*pair) for pair in zip(out_sock.types, in_sock.types))
    ]

    # No connections seem to be possible
    if not possible_connections:
        connections_status_str = _connections_status(
            from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
        )

        # Both sockets were specified: explain why the types don't match
        if len(from_sockets) == len(to_sockets) and len(from_sockets) == 1:
            raise PipelineConnectError(
                f"Cannot connect '{from_node}.{from_sockets[0].name}' with '{to_node}.{to_sockets[0].name}': "
                f"their declared input and output types do not match.\n{connections_status_str}"
            )

        # Not both sockets were specified: explain there's no possible match on any pair
        connections_status_str = _connections_status(
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
            connections_status_str = _connections_status(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )
            raise PipelineConnectError(
                f"Cannot connect '{from_node}' with '{to_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like: "
                f"pipeline.connect('{from_node}.{possible_connections[0][0].name}', "
                f"'{to_node}.{possible_connections[0][1].name}').\n{connections_status_str}"
            )

    return possible_connections[0]


def _connections_status(from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]):
    """
    Lists the status of the sockets, for error messages.
    """
    from_sockets_entries = []
    for from_socket in from_sockets:
        socket_types = ", ".join([_get_socket_type_desc(t) for t in from_socket.types])
        from_sockets_entries.append(f" - {from_socket.name} ({socket_types})")
    from_sockets_list = "\n".join(from_sockets_entries)

    to_sockets_entries = []
    for to_socket in to_sockets:
        socket_types = ", ".join([_get_socket_type_desc(t) for t in to_socket.types])
        to_sockets_entries.append(
            f" - {to_socket.name} ({socket_types}), {'sent by '+to_socket.sender if to_socket.sender else 'available'}"
        )
    to_sockets_list = "\n".join(to_sockets_entries)

    return f"'{from_node}':\n{from_sockets_list}\n'{to_node}':\n{to_sockets_list}"


def _get_socket_type_desc(type_):
    """
    Assembles a readable representation of the type of a connection. Can handle primitive types, classes, and
    arbitrarily nested structures of types from the typing module.
    """
    # get_args returns something only if this type has subtypes, in which case it needs to be printed differently.
    args = get_args(type_)

    if not args:
        if isinstance(type_, type):
            if type_.__name__.startswith("typing."):
                return type_.__name__[len("typing.") :]
            return type_.__name__

        # Literals only accept instances, not classes, so we need to account for those.
        if isinstance(type_, str):
            return f"'{type_}'"  # Quote strings
        if str(type_).startswith("typing."):
            return str(type_)[len("typing.") :]
        return str(type_)

    # Python < 3.10 support
    if hasattr(type_, "_name"):
        type_name = type_._name  # pylint: disable=protected-access
        # Support for Optionals and Unions in Python < 3.10
        if not type_name:
            if type(None) in args:
                type_name = "Optional"
                args = [a for a in args if a is not type(None)]
            else:
                if not any(isinstance(a, type) for a in args):
                    type_name = "Literal"
                else:
                    type_name = "Union"
    else:
        type_name = type_.__name__

    subtypes = ", ".join([_get_socket_type_desc(subtype) for subtype in args if subtype is not type(None)])
    return f"{type_name}[{subtypes}]"
