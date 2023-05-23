import re
from dataclasses import dataclass

import pytest

from canals.errors import PipelineConnectError
from canals.component import component, ComponentInput, ComponentOutput
from canals.pipeline.sockets import find_input_sockets, find_output_sockets
from canals.pipeline.connections import find_unambiguous_connection


def test_find_unambiguous_connection_no_connection_possible():
    @component
    class Component1:
        @dataclass
        class Input(ComponentInput):
            input_value: int

        @dataclass
        class Output(ComponentOutput):
            output_value: int

        def run(self, data: Input) -> Output:
            return Component1.Output(output_value=data.input_value)

    @component
    class Component2:
        @dataclass
        class Input(ComponentInput):
            input_value: str

        @dataclass
        class Output(ComponentOutput):
            output_value: str

        def run(self, data: Input) -> Output:
            return Component2.Output(output_value=data.input_value)

    expected_message = """Cannot connect 'comp1' with 'comp2': no matching connections available.
'comp1':
 - output_value \(int\)
'comp2':
 - input_value \(str, available\)"""

    with pytest.raises(PipelineConnectError, match=expected_message):
        find_unambiguous_connection(
            from_node="comp1",
            to_node="comp2",
            from_sockets=find_output_sockets(Component1()).values(),
            to_sockets=find_input_sockets(Component2()).values(),
        )


def test_find_unambiguous_connection_many_connections_possible_name_matches():
    @component
    class Component1:
        @dataclass
        class Input(ComponentInput):
            value: str

        @dataclass
        class Output(ComponentOutput):
            value: str

        def run(self, data: Input) -> Output:
            return Component1.Output(value=data.value)

    @component
    class Component2:
        @dataclass
        class Input(ComponentInput):
            value: str
            othervalue: str
            yetanothervalue: str

        @dataclass
        class Output(ComponentOutput):
            value: str

        def run(self, data: Input) -> Output:
            return Component2.Output(value=data.value)

    comp1 = Component1()
    comp2 = Component2()
    connection = find_unambiguous_connection(
        from_node="comp1",
        to_node="comp2",
        from_sockets=find_output_sockets(comp1).values(),
        to_sockets=find_input_sockets(comp2).values(),
    )
    assert connection == (find_output_sockets(comp1)["value"], find_input_sockets(comp2)["value"])


def test_find_unambiguous_connection_many_connections_possible_no_name_matches():
    @component
    class Component1:
        @dataclass
        class Input(ComponentInput):
            value: str

        @dataclass
        class Output(ComponentOutput):
            value: str

        def run(self, data: Input) -> Output:
            return Component1.Output(value=data.value)

    @component
    class Component2:
        @dataclass
        class Input(ComponentInput):
            value1: str
            value2: str
            value3: str

        @dataclass
        class Output(ComponentOutput):
            value: str

        def run(self, data: Input) -> Output:
            return Component2.Output(value=data.value1)

    expected_message = """Cannot connect 'comp1' with 'comp2': more than one connection is possible between these components. Please specify the connection name, like: pipeline.connect\('comp1.value', 'comp2.value1'\).
'comp1':
 - value \(str\)
'comp2':
 - value1 \(str, available\)
 - value2 \(str, available\)
 - value3 \(str, available\)"""
    with pytest.raises(PipelineConnectError, match=expected_message):
        find_unambiguous_connection(
            from_node="comp1",
            to_node="comp2",
            from_sockets=find_output_sockets(Component1()).values(),
            to_sockets=find_input_sockets(Component2()).values(),
        )
