# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Set, Sequence, Tuple, Dict, Mapping, Literal, Union, Optional, Any
from enum import Enum
import re
from pathlib import Path

import pytest

from canals.errors import PipelineConnectError
from canals.pipeline import Pipeline, PipelineConnectError
from canals.component import component
from canals.pipeline.sockets import find_input_sockets, find_output_sockets
from canals.pipeline.connections import find_unambiguous_connection

from test.sample_components import AddFixedValue, Greet
from test._helpers.component_factory import make_component


class TestClass1:
    ...


class TestClass2:
    ...


class TestClass3(TestClass1):
    ...


class TestEnum(Enum):
    TEST1 = TestClass1
    TEST2 = TestClass2


@pytest.mark.parametrize(
    "output,input",
    [
        pytest.param(str, str, id="same-primitives"),
        pytest.param(str, Optional[str], id="receiving-primitive-is-optional"),
        pytest.param(Optional[str], str, id="sending-primitive-is-optional"),
        # pytest.param(str, Any, id="primitive-to-any"),
        pytest.param(TestClass1, TestClass1, id="same-class"),
        pytest.param(TestClass1, Optional[TestClass1], id="receiving-class-is-optional"),
        pytest.param(Optional[TestClass1], TestClass1, id="sending-class-is-optional"),
        # pytest.param(TestClass1, TestClass1, id="class-to-any"),
        # pytest.param(TestClass3, TestClass1, id="subclass-to-class"),
        pytest.param(List[int], List[int], id="same-lists"),
        pytest.param(List[int], Optional[List[int]], id="receiving-list-is-optional"),
        pytest.param(Optional[List[int]], List[int], id="sending-list-is-optional"),
        # pytest.param(List[int], List[Any], id="list-of-primitive-to-list-of-any"),
        pytest.param(List[TestClass1], List[TestClass1], id="list-of-same-classes"),
        # pytest.param(List[TestClass3], List[TestClass1], id="list-of-subclass-to-list-of-class"),
        # pytest.param(List[TestClass1], List[Any], id="list-of-classes-to-list-of-any"),
        pytest.param(List[Set[Sequence[bool]]], List[Set[Sequence[bool]]], id="nested-sequences-of-same-primitives"),
        # pytest.param(
        #     List[Set[Sequence[bool]]],
        #     List[Set[Sequence[Any]]],
        #     id="nested-sequences-of-primitives-to-nested-sequences-of-any",
        # ),
        pytest.param(
            List[Set[Sequence[TestClass1]]], List[Set[Sequence[TestClass1]]], id="nested-sequences-of-same-classes"
        ),
        # pytest.param(
        #     List[Set[Sequence[TestClass3]]],
        #     List[Set[Sequence[TestClass1]]],
        #     id="nested-sequences-of-subclasses-to-nested-sequences-of-classes",
        # ),
        # pytest.param(
        #     List[Set[Sequence[TestClass1]]],
        #     List[Set[Sequence[Any]]],
        #     id="nested-sequences-of-classes-to-nested-sequences-of-any",
        # ),
        pytest.param(Dict[str, int], Dict[str, int], id="same-dicts-of-primitives"),
        # pytest.param(Dict[str, int], Dict[Any, int], id="dict-of-primitives-to-dict-of-any-keys"),
        # pytest.param(Dict[str, int], Dict[str, Any], id="dict-of-primitives-to-dict-of-any-values"),
        # pytest.param(Dict[str, int], Dict[Any, Any], id="dict-of-primitives-to-dict-of-any-key-and-values"),
        pytest.param(Dict[str, TestClass1], Dict[str, TestClass1], id="same-dicts-of-classes-values"),
        # pytest.param(Dict[str, TestClass3], Dict[str, TestClass1], id="dict-of-subclasses-to-dict-of-classes"),
        # pytest.param(Dict[str, TestClass1], Dict[Any, TestClass1], id="dict-of-classes-to-dict-of-any-keys"),
        # pytest.param(Dict[str, TestClass1], Dict[str, Any], id="dict-of-classes-to-dict-of-any-values"),
        # pytest.param(Dict[str, TestClass1], Dict[Any, Any], id="dict-of-classes-to-dict-of-any-key-and-values"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mappings-of-same-primitives",
        ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, int]]],
        #     Dict[str, Mapping[str, Dict[Any, int]]],
        #     id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, int]]],
        #     Dict[str, Mapping[Any, Dict[str, int]]],
        #     id="nested-mapping-of-primitives-to-nested-mapping-of-higher-level-any-keys",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, int]]],
        #     Dict[str, Mapping[str, Dict[str, Any]]],
        #     id="nested-mapping-of-primitives-to-nested-mapping-of-any-values",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, int]]],
        #     Dict[str, Mapping[Any, Dict[Any, Any]]],
        #     id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys-and-values",
        # ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            id="nested-mappings-of-same-classes",
        ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, TestClass3]]],
        #     Dict[str, Mapping[str, Dict[str, TestClass1]]],
        #     id="nested-mapping-of-subclasses-to-nested-mapping-of-classes",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, TestClass1]]],
        #     Dict[str, Mapping[str, Dict[Any, TestClass1]]],
        #     id="nested-mapping-of-classes-to-nested-mapping-of-any-keys",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, TestClass1]]],
        #     Dict[str, Mapping[Any, Dict[str, TestClass1]]],
        #     id="nested-mapping-of-classes-to-nested-mapping-of-higher-level-any-keys",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, TestClass1]]],
        #     Dict[str, Mapping[str, Dict[str, Any]]],
        #     id="nested-mapping-of-classes-to-nested-mapping-of-any-values",
        # ),
        # pytest.param(
        #     Dict[str, Mapping[str, Dict[str, TestClass1]]],
        #     Dict[str, Mapping[Any, Dict[Any, Any]]],
        #     id="nested-mapping-of-classes-to-nested-mapping-of-any-keys-and-values",
        # ),
        pytest.param(
            Literal["a", "b", "c"],
            Literal["a", "b", "c"],
            id="same-primitive-literal",
        ),
        pytest.param(
            Literal[TestEnum.TEST1],
            Literal[TestEnum.TEST1],
            id="same-enum-literal",
        ),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass1]]],
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass1]]],
            id="identical-deeply-nested-complex-type",
        ),
    ],
)
def test_connect_compatible_types(output, input):
    comp1 = make_component(output=output)
    comp2 = make_component(input=input)

    pipe = Pipeline()
    pipe.add_component("c1", comp1)
    pipe.add_component("c2", comp2)
    pipe.connect("c1", "c2")
    assert list(pipe.graph.edges) == [("c1", "c2", "value/value")]


@pytest.mark.parametrize(
    "input,output",
    [
        pytest.param(int, bool, id="different-primitives"),
        pytest.param(TestClass1, TestClass2, id="different-classes"),
        pytest.param(TestClass1, TestClass3, id="class-to-subclass"),
        # pytest.param(Any, int, id="any-to-primitive"),
        # pytest.param(Any, TestClass2, id="any-to-class"),
        pytest.param(List[int], List[str], id="different-lists-of-primitives"),
        pytest.param(List[TestClass1], List[TestClass2], id="different-lists-of-classes"),
        pytest.param(List[TestClass1], List[TestClass3], id="lists-of-classes-to-subclasses"),
        pytest.param(List[Any], List[str], id="list-of-any-to-list-of-primitives"),
        pytest.param(List[Any], List[TestClass2], id="list-of-any-to-list-of-classes"),
        pytest.param(
            List[Set[Sequence[str]]], List[Set[Sequence[bool]]], id="nested-sequences-of-different-primitives"
        ),
        pytest.param(
            List[Set[Sequence[str]]], Set[List[Sequence[str]]], id="different-nested-sequences-of-same-primitives"
        ),
        pytest.param(
            List[Set[Sequence[TestClass1]]], List[Set[Sequence[TestClass2]]], id="nested-sequences-of-different-classes"
        ),
        pytest.param(
            List[Set[Sequence[TestClass1]]],
            List[Set[Sequence[TestClass3]]],
            id="nested-sequences-of-classes-to-subclasses",
        ),
        pytest.param(
            List[Set[Sequence[TestClass1]]],
            Set[List[Sequence[TestClass1]]],
            id="different-nested-sequences-of-same-class",
        ),
        pytest.param(
            List[Set[Sequence[Any]]],
            List[Set[Sequence[bool]]],
            id="nested-list-of-Any-to-nested-list-of-primitives",
        ),
        pytest.param(
            List[Set[Sequence[Any]]],
            List[Set[Sequence[TestClass2]]],
            id="nested-list-of-Any-to-nested-list-of-classes",
        ),
        pytest.param(Dict[str, int], Dict[int, int], id="different-dict-of-primitive-keys"),
        pytest.param(Dict[str, int], Dict[str, bool], id="different-dict-of-primitive-values"),
        pytest.param(Dict[str, TestClass1], Dict[str, TestClass2], id="different-dict-of-class-values"),
        pytest.param(Dict[str, TestClass1], Dict[str, TestClass3], id="different-dict-of-class-to-subclass-values"),
        pytest.param(Dict[Any, int], Dict[int, int], id="dict-of-Any-keys-to-dict-of-primitives"),
        pytest.param(Dict[str, Any], Dict[int, int], id="dict-of-Any-values-to-dict-of-primitives"),
        pytest.param(Dict[str, Any], Dict[int, TestClass1], id="dict-of-Any-values-to-dict-of-classes"),
        pytest.param(Dict[Any, Any], Dict[int, int], id="dict-of-Any-keys-and-values-to-dict-of-primitives"),
        pytest.param(Dict[Any, Any], Dict[int, TestClass1], id="dict-of-Any-keys-and-values-to-dict-of-classes"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Mapping[str, Dict[str, Dict[str, int]]],
            id="different-nested-mappings-of-same-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[int, int]]],
            id="same-nested-mappings-of-different-primitive-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[int, Dict[str, int]]],
            id="same-nested-mappings-of-different-higer-level-primitive-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[str, bool]]],
            id="same-nested-mappings-of-different-primitive-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            Dict[str, Mapping[str, Dict[str, TestClass2]]],
            id="same-nested-mappings-of-different-class-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            Dict[str, Mapping[str, Dict[str, TestClass2]]],
            id="same-nested-mappings-of-class-to-subclass-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[Any, int]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mapping-of-Any-keys-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[Any, Dict[Any, int]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mapping-of-higher-level-Any-keys-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Any]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mapping-of-Any-values-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Any]]],
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            id="nested-mapping-of-Any-values-to-nested-mapping-of-classes",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[Any, Any]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[Any, Any]]],
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-classes",
        ),
        pytest.param(
            Literal["a", "b", "c"],
            Literal["a", "b"],
            id="different-literal-of-same-primitive",
        ),
        pytest.param(
            Literal[TestEnum.TEST1],
            Literal[TestEnum.TEST2],
            id="different-literal-of-same-enum",
        ),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass1]]],
            Tuple[Literal["a", "b", "c"], Union[Path, Dict[int, TestClass1]]],
            id="deeply-nested-complex-type-is-compatible-but-cannot-be-checked",
        ),
    ],
)
def test_connect_non_compatible_types(output, input):
    comp1 = make_component(output=output)
    comp2 = make_component(input=input)

    pipe = Pipeline()
    pipe.add_component("c1", comp1)
    pipe.add_component("c2", comp2)

    with pytest.raises(PipelineConnectError, match="Cannot connect 'c1' with 'c2': no matching connections available."):
        pipe.connect("c1", "c2")


def test_connect_nonexisting_from_component():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("third", "second")


def test_connect_nonexisting_to_component():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("first", "third")


def test_connect_nonexisting_from_socket():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="first.wrong does not exist"):
        pipe.connect("first.wrong", "second")


def test_connect_nonexisting_to_socket():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="second.wrong does not exist"):
        pipe.connect("first", "second.wrong")


def test_connect_mismatched_components():
    add = AddFixedValue()
    greet = Greet()

    pipe = Pipeline()
    pipe.add_component("first", add)
    pipe.add_component("second", greet)
    with pytest.raises(
        PipelineConnectError, match="Cannot connect 'first' with 'second': no matching connections available."
    ):
        pipe.connect("first", "second.message")


def test_connect_many_outputs_to_the_same_input():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    pipe.add_component("third", add_2)
    pipe.connect("first.value", "second.value")
    with pytest.raises(PipelineConnectError, match="second.value is already connected to first"):
        pipe.connect("third.value", "second.value")


def test_find_unambiguous_connection_no_connection_possible():
    @component
    class Component1:
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

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                input_value: str

            return Input

        @component.output
        def output(self):
            class Output:
                output_value: str

            return Output

        def run(self, data):
            return self.output(output_value=data.input_value)

    expected_message = re.escape(
        """Cannot connect 'comp1' with 'comp2': no matching connections available.
'comp1':
 - output_value (int)
'comp2':
 - input_value (str), available"""
    )

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
        @component.input
        def input(self):
            class Input:
                value: str

            return Input

        @component.output
        def output(self):
            class Output:
                value: str

            return Output

        def run(self, data):
            return self.output(value=data.value)

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: str
                othervalue: str
                yetanothervalue: str

            return Input

        @component.output
        def output(self):
            class Output:
                value: str

            return Output

        def run(self, data):
            return self.output(value=data.value)

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
        @component.input
        def input(self):
            class Input:
                value: str

            return Input

        @component.output
        def output(self):
            class Output:
                value: str

            return Output

        def run(self, data):
            return self.output(value=data.value)

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value1: str
                value2: str
                value3: str

            return Input

        @component.output
        def output(self):
            class Output:
                value: str

            return Output

        def run(self, data):
            return self.output(value=data.value1)

    expected_message = re.escape(
        """Cannot connect 'comp1' with 'comp2': more than one connection is possible between these components. Please specify the connection name, like: pipeline.connect('comp1.value', 'comp2.value1').
'comp1':
 - value (str)
'comp2':
 - value1 (str), available
 - value2 (str), available
 - value3 (str), available"""
    )
    with pytest.raises(PipelineConnectError, match=expected_message):
        find_unambiguous_connection(
            from_node="comp1",
            to_node="comp2",
            from_sockets=find_output_sockets(Component1()).values(),
            to_sockets=find_input_sockets(Component2()).values(),
        )
