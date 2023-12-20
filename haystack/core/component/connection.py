import itertools
from typing import Optional, List, Tuple
from dataclasses import dataclass

from haystack.core.component.sockets import InputSocket, OutputSocket
from haystack.core.type_utils import _type_name, _types_are_compatible
from haystack.core.errors import PipelineConnectError


@dataclass
class Connection:
    sender: Optional[str]
    sender_socket: Optional[OutputSocket]
    receiver: Optional[str]
    receiver_socket: Optional[InputSocket]

    def __post_init__(self):
        if self.sender and self.sender_socket and self.receiver and self.receiver_socket:
            # If we're trying to connect the same sockets again, this should be a no-op
            if self.receiver in self.sender_socket.receivers and self.sender in self.receiver_socket.senders:
                return

            # Make sure the receiving socket isn't already connected, unless it's variadic. Sending sockets can be
            # connected as many times as needed, so they don't need this check
            if self.receiver_socket.senders and not self.receiver_socket.is_variadic:
                raise PipelineConnectError(
                    f"Cannot connect '{self.sender}.{self.sender_socket.name}' with '{self.receiver}.{self.receiver_socket.name}': "
                    f"{self.receiver}.{self.receiver_socket.name} is already connected to {self.receiver_socket.senders}.\n"
                )

            self.sender_socket.receivers.append(self.receiver)
            self.receiver_socket.senders.append(self.sender)

    def __repr__(self):
        if self.sender and self.sender_socket:
            sender_repr = f"{self.sender}.{self.sender_socket.name} ({_type_name(self.sender_socket.type)})"
        else:
            sender_repr = "input needed"

        if self.receiver and self.receiver_socket:
            receiver_repr = f"({_type_name(self.receiver_socket.type)}) {self.receiver}.{self.receiver_socket.name}"
        else:
            receiver_repr = "output"

        return f"{sender_repr} --> {receiver_repr}"

    def __hash__(self):
        """
        Connection is used as a dictionary key in Pipeline, it must be hashable
        """
        return hash(
            "-".join(
                [
                    self.sender if self.sender else "input",
                    self.sender_socket.name if self.sender_socket else "",
                    self.receiver if self.receiver else "output",
                    self.receiver_socket.name if self.receiver_socket else "",
                ]
            )
        )

    @property
    def is_mandatory(self) -> bool:
        """
        Returns True if the connection goes to a mandatory input socket, False otherwise
        """
        if self.receiver_socket:
            return self.receiver_socket.is_mandatory
        return False

    @staticmethod
    def from_list_of_sockets(
        sender_node: str, sender_sockets: List[OutputSocket], receiver_node: str, receiver_sockets: List[InputSocket]
    ) -> "Connection":
        """
        Find one single possible connection between two lists of sockets.
        """
        # List all sender/receiver combinations of sockets that match by type
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
            if len(name_matches) == 1:
                # Sockets match by type and name, let's use this
                return Connection(sender_node, name_matches[0][0], receiver_node, name_matches[0][1])

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

        match = possible_connections[0]
        return Connection(sender_node, match[0], receiver_node, match[1])


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
        if receiver_socket.senders:
            sender_status = f"sent by {','.join(receiver_socket.senders)}"
        else:
            sender_status = "available"
        receiver_sockets_entries.append(
            f" - {receiver_socket.name}: {_type_name(receiver_socket.type)} ({sender_status})"
        )
    receiver_sockets_list = "\n".join(receiver_sockets_entries)

    return f"'{sender_node}':\n{sender_sockets_list}\n'{receiver_node}':\n{receiver_sockets_list}"


def parse_connect_string(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None
