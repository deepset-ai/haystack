# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Union, Set, Sequence, Iterable, Dict, Mapping, Tuple

from canals.component import component
from canals.pipeline.sockets import (
    find_output_sockets,
    OutputSocket,
)


def test_find_output_sockets_one_regular_builtin_type_output():
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
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {"output_value": OutputSocket(name="output_value", type=int)}
    assert sockets == expected


def test_find_output_sockets_many_regular_builtin_type_outputs():
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
                int_value: int
                str_value: str
                bool_value: bool

            return Output

        def run(self, data):
            return self.output(int_value=1, str_value="1", bool_value=True)

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "int_value": OutputSocket(name="int_value", type=int),
        "str_value": OutputSocket(name="str_value", type=str),
        "bool_value": OutputSocket(name="bool_value", type=bool),
    }
    assert sockets == expected


def test_find_output_sockets_one_regular_object_type_output():
    class MyObject:
        ...

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
                output_value: MyObject

            return Output

        def run(self, data):
            return self.output(output_value=MyObject())

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {"output_value": OutputSocket(name="output_value", type=MyObject)}
    assert sockets == expected


def test_find_output_sockets_one_union_type_output():
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
                output_value: Union[str, int]

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {"output_value": OutputSocket(name="output_value", type=Union[str, int])}
    assert sockets == expected


def test_find_output_sockets_one_optional_builtin_type_output():
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
                output_value: Optional[int] = None

            return Output

        def run(self, data):
            return self.output(output_value=1)

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {"output_value": OutputSocket(name="output_value", type=Optional[int])}
    assert sockets == expected


def test_find_output_sockets_one_optional_object_type_output():
    class MyObject:
        ...

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
                output_value: Optional[MyObject] = None

            return Output

        def run(self, data):
            return self.output(output_value=MyObject())

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {"output_value": OutputSocket(name="output_value", type=Optional[MyObject])}
    assert sockets == expected


def test_find_output_sockets_sequences_of_builtin_type_output():
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
                list_value: List[int]
                set_value: Set[int]
                sequence_value: Sequence[int]
                iterable_value: Iterable[int]

            return Output

        def run(self, data):
            return self.output(list_value=[], set_value=set(), sequence_value=[], iterable_value=[])

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "list_value": OutputSocket(name="list_value", type=List[int]),
        "set_value": OutputSocket(name="set_value", type=Set[int]),
        "sequence_value": OutputSocket(name="sequence_value", type=Sequence[int]),
        "iterable_value": OutputSocket(name="iterable_value", type=Iterable[int]),
    }
    assert sockets == expected


def test_find_output_sockets_sequences_of_object_type_output():
    class MyObject:
        ...

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
                list_value: List[MyObject]
                set_value: Set[MyObject]
                sequence_value: Sequence[MyObject]
                iterable_value: Iterable[MyObject]

            return Output

        def run(self, data):
            return self.output(list_value=[], set_value=set(), sequence_value=[], iterable_value=[])

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "list_value": OutputSocket(name="list_value", type=List[MyObject]),
        "set_value": OutputSocket(name="set_value", type=Set[MyObject]),
        "sequence_value": OutputSocket(name="sequence_value", type=Sequence[MyObject]),
        "iterable_value": OutputSocket(name="iterable_value", type=Iterable[MyObject]),
    }
    assert sockets == expected


def test_find_output_sockets_mappings_of_builtin_type_output():
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
                dict_value: Dict[str, int]
                mapping_value: Mapping[str, int]

            return Output

        def run(self, data):
            return self.output(dict_value={}, mapping_value={})

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "dict_value": OutputSocket(name="dict_value", type=Dict[str, int]),
        "mapping_value": OutputSocket(name="mapping_value", type=Mapping[str, int]),
    }
    assert sockets == expected


def test_find_output_sockets_mappings_of_object_type_output():
    class MyObject:
        ...

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
                dict_value: Dict[str, MyObject]
                mapping_value: Mapping[str, MyObject]

            return Output

        def run(self, data):
            return self.output(dict_value={}, mapping_value={})

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "dict_value": OutputSocket(name="dict_value", type=Dict[str, MyObject]),
        "mapping_value": OutputSocket(name="mapping_value", type=Mapping[str, MyObject]),
    }
    assert sockets == expected


def test_find_output_sockets_tuple_type_output():
    class MyObject:
        ...

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
                tuple_value: Tuple[str, MyObject]

            return Output

        def run(self, data):
            return self.output(tuple_value=("a", MyObject()))

    comp = MockComponent()
    sockets = find_output_sockets(comp)
    expected = {
        "tuple_value": OutputSocket(name="tuple_value", type=Tuple[str, MyObject]),
    }
    assert sockets == expected
