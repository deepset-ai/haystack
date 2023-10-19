# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, List

import logging
import itertools

from canals.errors import PipelineConnectError
from canals.type_utils import _types_are_compatible, _type_name
from canals.component.sockets import InputSocket, OutputSocket


logger = logging.getLogger(__name__)


def parse_connection(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def _find_unambiguous_connection(
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
        sender_sockets_entries.append(f" - {sender_socket.name}: {_type_name(sender_socket.type)}")
    sender_sockets_list = "\n".join(sender_sockets_entries)

    receiver_sockets_entries = []
    for receiver_socket in receiver_sockets:
        if receiver_socket.sender:
            sender_status = f"sent by {','.join(receiver_socket.sender)}"
        else:
            sender_status = "available"
        receiver_sockets_entries.append(
            f" - {receiver_socket.name}: {_type_name(receiver_socket.type)} ({sender_status})"
        )
    receiver_sockets_list = "\n".join(receiver_sockets_entries)

    return f"'{sender_node}':\n{sender_sockets_list}\n'{receiver_node}':\n{receiver_sockets_list}"
