# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Set, Sequence, Tuple, Dict, Mapping, Literal, Union, Optional
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


def test_connect():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    pipe.connect("first", "second")

    assert list(pipe.graph.edges) == [("first", "second", "value/value")]


def test_connect_on_primitive_types():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: int

            return Input

        @component.output
        def output(self):
            class Output:
                value: str

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: str

            return Input

        @component.output
        def output(self):
            class Output:
                value: bool

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (bool)
'comp1':
 - value (int), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_classes():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: TestClass1

            return Input

        @component.output
        def output(self):
            class Output:
                value: TestClass2

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: TestClass2

            return Input

        @component.output
        def output(self):
            class Output:
                value: TestClass3

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (TestClass3)
'comp1':
 - value (TestClass1), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_shallow_sequence_type_with_primitives():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: List[int]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[str]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: List[str]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[bool]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (List[bool])
'comp1':
 - value (List[int]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_deeply_nested_sequence_type_with_primitives():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: List[Set[Sequence[int]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[Set[Sequence[str]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: List[Set[Sequence[str]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[Set[Sequence[bool]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (List[Set[Sequence[bool]]])
'comp1':
 - value (List[Set[Sequence[int]]]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_shallow_sequence_type_with_classes():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: List[TestClass1]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[TestClass2]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: List[TestClass2]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[TestClass3]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (List[TestClass3])
'comp1':
 - value (List[TestClass1]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_deeply_nested_sequence_type_with_classes():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: List[Set[Sequence[TestClass1]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[Set[Sequence[TestClass2]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: List[Set[Sequence[TestClass2]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: List[Set[Sequence[TestClass3]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (List[Set[Sequence[TestClass3]]])
'comp1':
 - value (List[Set[Sequence[TestClass1]]]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_shallow_mapping_type_with_primitives():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, int]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, str]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, str]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, bool]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Dict[str, bool])
'comp1':
 - value (Dict[str, int]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_deeply_nested_mapping_type_with_primitives():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, Mapping[str, Dict[str, int]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, Mapping[str, Dict[str, str]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, Mapping[str, Dict[str, str]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, Mapping[str, Dict[str, bool]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Dict[str, Mapping[str, Dict[str, bool]]])
'comp1':
 - value (Dict[str, Mapping[str, Dict[str, int]]]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_shallow_mapping_type_with_classes():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, TestClass1]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, TestClass2]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, TestClass2]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, TestClass3]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Dict[str, TestClass3])
'comp1':
 - value (Dict[str, TestClass1]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_deeply_nested_mapping_type_with_classes():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, Mapping[str, Dict[str, TestClass1]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, Mapping[str, Dict[str, TestClass2]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Dict[str, Mapping[str, Dict[str, TestClass2]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Dict[str, Mapping[str, Dict[str, TestClass3]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Dict[str, Mapping[str, Dict[str, TestClass3]]])
'comp1':
 - value (Dict[str, Mapping[str, Dict[str, TestClass1]]]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_literal_with_primitives():
    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Literal["a", "b", "c"]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Literal["1", "2", "3"]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Literal["1", "2", "3"]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Literal["x", "y", "z"]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Literal['x', 'y', 'z'])
'comp1':
 - value (Literal['a', 'b', 'c']), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_literal_with_enums():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    class TestEnum(Enum):
        TEST1 = TestClass1
        TEST2 = TestClass2
        TEST3 = TestClass3

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Literal[TestEnum.TEST1]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Literal[TestEnum.TEST2]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Literal[TestEnum.TEST2]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Literal[TestEnum.TEST3]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Literal[TestEnum.TEST3])
'comp1':
 - value (Literal[TestEnum.TEST1]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_optional_with_primitives():
    """
    Note that Optionals are "transparent", so they disappear when top-level
    """

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Optional[int]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Optional[str]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Optional[str]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Optional[bool]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (bool)
'comp1':
 - value (int), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_optional_with_classes():
    """
    Note that Optionals are "transparent", so they disappear when top-level
    """

    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Optional[TestClass1]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Optional[TestClass2]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Optional[TestClass2]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Optional[TestClass3]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (TestClass3)
'comp1':
 - value (TestClass1), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


def test_connect_on_a_deeply_nested_complex_type():
    class TestClass1:
        ...

    class TestClass2:
        ...

    class TestClass3:
        ...

    @component
    class Component:
        @component.input
        def input(self):
            class Input:
                value: Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass1]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass2]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    @component
    class Component2:
        @component.input
        def input(self):
            class Input:
                value: Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass2]]]

            return Input

        @component.output
        def output(self):
            class Output:
                value: Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass3]]]

            return Output

        def run(self, data):
            return self.output(value="a")

    pipe = Pipeline()
    pipe.add_component("comp1", Component())
    pipe.add_component("comp2", Component2())
    pipe.connect("comp1", "comp2")

    with pytest.raises(
        PipelineConnectError,
        match=re.escape(
            """Cannot connect 'comp2' with 'comp1': no matching connections available.
'comp2':
 - value (Tuple[Optional[Literal['a', 'b', 'c']], Union[Path, Dict[int, TestClass3]]])
'comp1':
 - value (Tuple[Optional[Literal['a', 'b', 'c']], Union[Path, Dict[int, TestClass1]]]), available"""
        ),
    ):
        pipe.connect("comp2", "comp1")


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
