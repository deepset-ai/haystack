from haystack.core.component.connection import Connection
from haystack.core.component.sockets import InputSocket, OutputSocket
from haystack.core.errors import PipelineConnectError

import pytest


@pytest.mark.parametrize(
    "c,expected",
    [
        (
            Connection("source_component", OutputSocket("out", int), "destination_component", InputSocket("in", int)),
            "source_component.out (int) --> (int) destination_component.in",
        ),
        (
            Connection(None, None, "destination_component", InputSocket("in", int)),
            "input needed --> (int) destination_component.in",
        ),
        (Connection("source_component", OutputSocket("out", int), None, None), "source_component.out (int) --> output"),
        (Connection(None, None, None, None), "input needed --> output"),
    ],
)
def test_repr(c, expected):
    assert str(c) == expected


def test_is_mandatory():
    c = Connection(None, None, "destination_component", InputSocket("in", int))
    assert c.is_mandatory

    c = Connection(None, None, "destination_component", InputSocket("in", int, 42))
    assert not c.is_mandatory

    c = Connection("source_component", OutputSocket("out", int), None, None)
    assert not c.is_mandatory


def test_from_list_of_sockets():
    sender_sockets = [OutputSocket("out_int", int), OutputSocket("out_str", str)]

    receiver_sockets = [InputSocket("in_str", str)]

    c = Connection.from_list_of_sockets("from_node", sender_sockets, "to_node", receiver_sockets)
    assert c.sender_socket.name == "out_str"  # type:ignore


def test_from_list_of_sockets_not_possible():
    sender_sockets = [OutputSocket("out_int", int), OutputSocket("out_str", str)]

    receiver_sockets = [InputSocket("in_list", list), InputSocket("in_tuple", tuple)]

    with pytest.raises(PipelineConnectError, match="no matching connections available"):
        Connection.from_list_of_sockets("from_node", sender_sockets, "to_node", receiver_sockets)


def test_from_list_of_sockets_too_many():
    sender_sockets = [OutputSocket("out_int", int), OutputSocket("out_str", str)]

    receiver_sockets = [InputSocket("in_int", int), InputSocket("in_str", str)]

    with pytest.raises(PipelineConnectError, match="more than one connection is possible"):
        Connection.from_list_of_sockets("from_node", sender_sockets, "to_node", receiver_sockets)


def test_from_list_of_sockets_only_one():
    sender_sockets = [OutputSocket("out_int", int)]

    receiver_sockets = [InputSocket("in_str", str)]

    with pytest.raises(PipelineConnectError, match="their declared input and output types do not match"):
        Connection.from_list_of_sockets("from_node", sender_sockets, "to_node", receiver_sockets)
