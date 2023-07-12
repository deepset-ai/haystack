# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import typing
from typing import List, Optional, Union, Set, Sequence, Iterable, Dict, Mapping, Tuple

from canals.component import component
from canals.pipeline.sockets import (
    find_input_sockets,
    InputSocket,
)


def test_find_input_sockets_one_regular_builtin_type_input():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: int

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=data.input_value)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {"input_value": InputSocket(name="input_value", type=int)}
    assert sockets == expected


def test_find_input_sockets_many_regular_builtin_type_inputs():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                int_value: int
                str_value: str
                bool_value: bool

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=data.int_value)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "int_value": InputSocket(name="int_value", type=int),
        "str_value": InputSocket(name="str_value", type=str),
        "bool_value": InputSocket(name="bool_value", type=bool),
    }
    assert sockets == expected


def test_find_input_sockets_one_regular_object_type_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: MyObject

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {"input_value": InputSocket(name="input_value", type=MyObject)}
    assert sockets == expected


def test_find_input_sockets_one_union_type_input():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: Union[str, int]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {"input_value": InputSocket(name="input_value", type=Union[str, int])}
    assert sockets == expected


def test_find_input_sockets_one_optional_builtin_type_input():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: Optional[int] = None

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {"input_value": InputSocket(name="input_value", type=Optional[int])}
    assert sockets == expected


def test_find_input_sockets_one_optional_object_type_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                input_value: Optional[MyObject] = None

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {"input_value": InputSocket(name="input_value", type=Optional[MyObject])}
    assert sockets == expected


def test_find_input_sockets_sequences_of_builtin_type_input():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                list_value: List[int]
                set_value: Set[int]
                sequence_value: Sequence[int]
                iterable_value: Iterable[int]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "list_value": InputSocket(name="list_value", type=typing.List[int]),
        "set_value": InputSocket(name="set_value", type=typing.Set[int]),
        "sequence_value": InputSocket(name="sequence_value", type=typing.Sequence[int]),
        "iterable_value": InputSocket(name="iterable_value", type=typing.Iterable[int]),
    }
    assert sockets == expected


def test_find_input_sockets_sequences_of_object_type_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                list_value: List[MyObject]
                set_value: Set[MyObject]
                sequence_value: Sequence[MyObject]
                iterable_value: Iterable[MyObject]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "list_value": InputSocket(name="list_value", type=typing.List[MyObject]),
        "set_value": InputSocket(name="set_value", type=typing.Set[MyObject]),
        "sequence_value": InputSocket(name="sequence_value", type=typing.Sequence[MyObject]),
        "iterable_value": InputSocket(name="iterable_value", type=typing.Iterable[MyObject]),
    }
    assert sockets == expected


def test_find_input_sockets_mappings_of_builtin_type_input():
    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                dict_value: Dict[str, int]
                mapping_value: Mapping[str, int]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "dict_value": InputSocket(name="dict_value", type=typing.Dict[str, int]),
        "mapping_value": InputSocket(name="mapping_value", type=typing.Mapping[str, int]),
    }
    assert sockets == expected


def test_find_input_sockets_mappings_of_object_type_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                dict_value: Dict[str, MyObject]
                mapping_value: Mapping[str, MyObject]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "dict_value": InputSocket(name="dict_value", type=typing.Dict[str, MyObject]),
        "mapping_value": InputSocket(name="mapping_value", type=typing.Mapping[str, MyObject]),
    }
    assert sockets == expected


def test_find_input_sockets_tuple_type_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input
        def input(self):
            class Input:
                tuple_value: Tuple[str, MyObject]

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: int

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_input_sockets(comp)
    expected = {
        "tuple_value": InputSocket(name="tuple_value", type=typing.Tuple[str, MyObject]),
    }
    assert sockets == expected
