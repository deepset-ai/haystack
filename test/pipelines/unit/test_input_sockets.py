import typing
from typing import List, Optional, Union, Set, Sequence, Iterable, Dict, Mapping, Tuple

from dataclasses import dataclass

import pytest

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
    expected = {"input_value": InputSocket(name="input_value", type=int, variadic=False)}
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
        "int_value": InputSocket(name="int_value", type=int, variadic=False),
        "str_value": InputSocket(name="str_value", type=str, variadic=False),
        "bool_value": InputSocket(name="bool_value", type=bool, variadic=False),
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
    expected = {"input_value": InputSocket(name="input_value", type=MyObject, variadic=False)}
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
    with pytest.raises(ValueError, match="Components do not support Union types for connections"):
        find_input_sockets(comp)


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
    expected = {"input_value": InputSocket(name="input_value", type=int, variadic=False)}
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
    expected = {"input_value": InputSocket(name="input_value", type=MyObject, variadic=False)}
    assert sockets == expected


def test_find_input_sockets_sequences_of_builtin_type_input_non_variadic():
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
        "list_value": InputSocket(name="list_value", type=typing.List[int], variadic=False),
        "set_value": InputSocket(name="set_value", type=typing.Set[int], variadic=False),
        "sequence_value": InputSocket(name="sequence_value", type=typing.Sequence[int], variadic=False),
        "iterable_value": InputSocket(name="iterable_value", type=typing.Iterable[int], variadic=False),
    }
    assert sockets == expected


def test_find_input_sockets_sequences_of_object_type_input_non_variadic():
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
        "list_value": InputSocket(name="list_value", type=typing.List[MyObject], variadic=False),
        "set_value": InputSocket(name="set_value", type=typing.Set[MyObject], variadic=False),
        "sequence_value": InputSocket(name="sequence_value", type=typing.Sequence[MyObject], variadic=False),
        "iterable_value": InputSocket(name="iterable_value", type=typing.Iterable[MyObject], variadic=False),
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
        "dict_value": InputSocket(name="dict_value", type=typing.Dict[str, int], variadic=False),
        "mapping_value": InputSocket(name="mapping_value", type=typing.Mapping[str, int], variadic=False),
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
        "dict_value": InputSocket(name="dict_value", type=typing.Dict[str, MyObject], variadic=False),
        "mapping_value": InputSocket(name="mapping_value", type=typing.Mapping[str, MyObject], variadic=False),
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
        "tuple_value": InputSocket(name="tuple_value", type=typing.Tuple[str, MyObject], variadic=False),
    }
    assert sockets == expected


def test_find_input_sockets_one_variadic_builtin_input():
    @component
    class MockComponent:
        @component.input(variadic=True)
        def input(self):
            class Input:
                values: List[int]

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
        "values": InputSocket(name="values", type=int, variadic=True),
    }
    assert sockets == expected


def test_find_input_sockets_variadic_object_input():
    class MyObject:
        ...

    @component
    class MockComponent:
        @component.input(variadic=True)
        def input(self):
            class Input:
                values: List[MyObject]

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
        "values": InputSocket(name="values", type=MyObject, variadic=True),
    }
    assert sockets == expected
