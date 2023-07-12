# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, Union, List, Any, get_args, get_origin

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


def _types_are_compatible(sender, receiver):  # pylint: disable=too-many-return-statements
    """
    Checks whether the source type is equal or a subtype of the destination type. Used to validate pipeline connections.

    Note: this method has no pretense to perform proper type matching. It especially does not deal with aliasing of
    typing classes such as `List` or `Dict` to their runtime counterparts `list` and `dict`. It also does not deal well
    with "bare" types, so `List` is treated differently from `List[Any]`, even though they should be the same.

    Consider simplifying the typing of your components if you observe unexpected errors during component connection.
    """
    if sender == receiver or receiver is Any:
        return True

    if sender is Any:
        return False

    try:
        if issubclass(sender, receiver):
            return True
    except TypeError:  # typing classes can't be used with issubclass, so we deal with them below
        pass

    sender_origin = get_origin(sender)
    receiver_origin = get_origin(receiver)

    if sender_origin is not Union and receiver_origin is Union:
        return any(_types_are_compatible(sender, union_arg) for union_arg in get_args(receiver))

    if not sender_origin or not receiver_origin or sender_origin != receiver_origin:
        return False

    sender_args = get_args(sender)
    receiver_args = get_args(receiver)
    if len(sender_args) > len(receiver_args):
        return False

    return all(_types_are_compatible(*args) for args in zip(sender_args, receiver_args))


def find_unambiguous_connection(
    sender_node: str, receiver_node: str, sender_sockets: List[OutputSocket], receiver_sockets: List[InputSocket]
) -> Tuple[OutputSocket, InputSocket]:
    """
    Find one single possible connection between two lists of sockets.
    """
    # List all combinations of sockets that match by type
    possible_connections = [
        (sender_sock, receiver_sock)
        for sender_sock, receiver_sock in itertools.product(sender_sockets, receiver_sockets)
        if _types_are_compatible(sender_sock.type, receiver_sock.type)
    ]

    # No connections seem to be possible
    if not possible_connections:
        connections_status_str = _connections_status(
            sender_node=sender_node,
            sender_sockets=sender_sockets,
            receiver_node=receiver_node,
            receiver_sockets=receiver_sockets,
        )

        # Both sockets were specified: explain why the types don't match
        if len(sender_sockets) == len(receiver_sockets) and len(sender_sockets) == 1:
            raise PipelineConnectError(
                f"Cannot connect '{sender_node}.{sender_sockets[0].name}' with '{receiver_node}.{receiver_sockets[0].name}': "
                f"their declared input and output types do not match.\n{connections_status_str}"
            )

        # Not both sockets were specified: explain there's no possible match on any pair
        connections_status_str = _connections_status(
            sender_node=sender_node,
            sender_sockets=sender_sockets,
            receiver_node=receiver_node,
            receiver_sockets=receiver_sockets,
        )
        raise PipelineConnectError(
            f"Cannot connect '{sender_node}' with '{receiver_node}': "
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
                sender_node=sender_node,
                sender_sockets=sender_sockets,
                receiver_node=receiver_node,
                receiver_sockets=receiver_sockets,
            )
            raise PipelineConnectError(
                f"Cannot connect '{sender_node}' with '{receiver_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like: "
                f"pipeline.connect('{sender_node}.{possible_connections[0][0].name}', "
                f"'{receiver_node}.{possible_connections[0][1].name}').\n{connections_status_str}"
            )

    return possible_connections[0]


def _connections_status(
    sender_node: str, receiver_node: str, sender_sockets: List[OutputSocket], receiver_sockets: List[InputSocket]
):
    """
    Lists the status of the sockets, for error messages.
    """
    sender_sockets_entries = []
    for sender_socket in sender_sockets:
        socket_types = get_socket_type_desc(sender_socket.type)
        sender_sockets_entries.append(f" - {sender_socket.name} ({socket_types})")
    sender_sockets_list = "\n".join(sender_sockets_entries)

    receiver_sockets_entries = []
    for receiver_socket in receiver_sockets:
        socket_types = get_socket_type_desc(receiver_socket.type)
        receiver_sockets_entries.append(
            f" - {receiver_socket.name} ({socket_types}), "
            f"{'sent by '+receiver_socket.sender if receiver_socket.sender else 'available'}"
        )
    receiver_sockets_list = "\n".join(receiver_sockets_entries)

    return f"'{sender_node}':\n{sender_sockets_list}\n'{receiver_node}':\n{receiver_sockets_list}"


def get_socket_type_desc(type_):
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

    subtypes = ", ".join([get_socket_type_desc(subtype) for subtype in args if subtype is not type(None)])
    return f"{type_name}[{subtypes}]"
