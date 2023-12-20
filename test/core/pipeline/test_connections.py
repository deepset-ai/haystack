# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Set, Sequence, Tuple, Dict, Mapping, Literal, Union, Optional, Any
from enum import Enum
import re
from pathlib import Path

import pytest

from haystack.core.pipeline import Pipeline
from haystack.core.errors import PipelineConnectError
from haystack.testing import factory
from haystack.core.component.connection import parse_connect_string
from haystack.testing.sample_components import AddFixedValue


class Class1:
    ...


class Class2:
    ...


class Class3(Class1):
    ...


class Enum1(Enum):
    TEST1 = Class1
    TEST2 = Class2


@pytest.mark.parametrize(
    "from_type,to_type",
    [
        pytest.param(str, str, id="same-primitives"),
        pytest.param(str, Optional[str], id="receiving-primitive-is-optional"),
        pytest.param(str, Union[int, str], id="receiving-type-is-union-of-primitives"),
        pytest.param(Union[int, str], Union[int, str], id="identical-unions"),
        pytest.param(Union[int, str], Union[int, str, bool], id="receiving-union-is-superset-of-sender"),
        pytest.param(str, Any, id="primitive-to-any"),
        pytest.param(Class1, Class1, id="same-class"),
        pytest.param(Class1, Optional[Class1], id="receiving-class-is-optional"),
        pytest.param(Class1, Class1, id="class-to-any"),
        pytest.param(Class3, Class1, id="subclass-to-class"),
        pytest.param(Class1, Union[int, Class1], id="receiving-type-is-union-of-classes"),
        pytest.param(Class3, Union[int, Class1], id="receiving-type-is-union-of-superclasses"),
        pytest.param(List[int], List[int], id="same-lists"),
        pytest.param(List[int], Optional[List[int]], id="receiving-list-is-optional"),
        pytest.param(List[int], List[Any], id="list-of-primitive-to-list-of-any"),
        pytest.param(List[Class1], List[Class1], id="list-of-same-classes"),
        pytest.param(List[Class3], List[Class1], id="list-of-subclass-to-list-of-class"),
        pytest.param(List[Class1], List[Any], id="list-of-classes-to-list-of-any"),
        pytest.param(List[Set[Sequence[bool]]], List[Set[Sequence[bool]]], id="nested-sequences-of-same-primitives"),
        pytest.param(
            List[Set[Sequence[bool]]],
            List[Set[Sequence[Any]]],
            id="nested-sequences-of-primitives-to-nested-sequences-of-any",
        ),
        pytest.param(List[Set[Sequence[Class1]]], List[Set[Sequence[Class1]]], id="nested-sequences-of-same-classes"),
        pytest.param(
            List[Set[Sequence[Class3]]],
            List[Set[Sequence[Class1]]],
            id="nested-sequences-of-subclasses-to-nested-sequences-of-classes",
        ),
        pytest.param(
            List[Set[Sequence[Class1]]],
            List[Set[Sequence[Any]]],
            id="nested-sequences-of-classes-to-nested-sequences-of-any",
        ),
        pytest.param(Dict[str, int], Dict[str, int], id="same-dicts-of-primitives"),
        pytest.param(Dict[str, int], Dict[Any, int], id="dict-of-primitives-to-dict-of-any-keys"),
        pytest.param(Dict[str, int], Dict[str, Any], id="dict-of-primitives-to-dict-of-any-values"),
        pytest.param(Dict[str, int], Dict[Any, Any], id="dict-of-primitives-to-dict-of-any-key-and-values"),
        pytest.param(Dict[str, Class1], Dict[str, Class1], id="same-dicts-of-classes-values"),
        pytest.param(Dict[str, Class3], Dict[str, Class1], id="dict-of-subclasses-to-dict-of-classes"),
        pytest.param(Dict[str, Class1], Dict[Any, Class1], id="dict-of-classes-to-dict-of-any-keys"),
        pytest.param(Dict[str, Class1], Dict[str, Any], id="dict-of-classes-to-dict-of-any-values"),
        pytest.param(Dict[str, Class1], Dict[Any, Any], id="dict-of-classes-to-dict-of-any-key-and-values"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mappings-of-same-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[Any, int]]],
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[Any, Dict[str, int]]],
            id="nested-mapping-of-primitives-to-nested-mapping-of-higher-level-any-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[str, Dict[str, Any]]],
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            Dict[str, Mapping[Any, Dict[Any, Any]]],
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys-and-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[str, Dict[str, Class1]]],
            id="nested-mappings-of-same-classes",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class3]]],
            Dict[str, Mapping[str, Dict[str, Class1]]],
            id="nested-mapping-of-subclasses-to-nested-mapping-of-classes",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[str, Dict[Any, Class1]]],
            id="nested-mapping-of-classes-to-nested-mapping-of-any-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[Any, Dict[str, Class1]]],
            id="nested-mapping-of-classes-to-nested-mapping-of-higher-level-any-keys",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[str, Dict[str, Any]]],
            id="nested-mapping-of-classes-to-nested-mapping-of-any-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[Any, Dict[Any, Any]]],
            id="nested-mapping-of-classes-to-nested-mapping-of-any-keys-and-values",
        ),
        pytest.param(Literal["a", "b", "c"], Literal["a", "b", "c"], id="same-primitive-literal"),
        pytest.param(Literal[Enum1.TEST1], Literal[Enum1.TEST1], id="same-enum-literal"),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, Class1]]],
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, Class1]]],
            id="identical-deeply-nested-complex-type",
        ),
    ],
)
def test_connect_compatible_types(from_type, to_type):
    comp1 = factory.component_class("Comp1", output_types={"value": from_type})()
    comp2 = factory.component_class("Comp2", input_types={"value": to_type})()

    pipe = Pipeline()
    pipe.add_component("c1", comp1)
    pipe.add_component("c2", comp2)
    pipe.connect("c1", "c2")
    assert list(pipe.graph.edges) == [("c1", "c2", "value/value")]


@pytest.mark.parametrize(
    "from_type, to_type",
    [
        pytest.param(int, bool, id="different-primitives"),
        pytest.param(Class1, Class2, id="different-classes"),
        pytest.param(Class1, Class3, id="class-to-subclass"),
        pytest.param(Any, int, id="any-to-primitive"),
        pytest.param(Any, Class2, id="any-to-class"),
        pytest.param(Optional[str], str, id="sending-primitive-is-optional"),
        pytest.param(Optional[Class1], Class1, id="sending-class-is-optional"),
        pytest.param(Optional[List[int]], List[int], id="sending-list-is-optional"),
        pytest.param(Union[int, str], str, id="sending-type-is-union"),
        pytest.param(Union[int, str, bool], Union[int, str], id="sending-union-is-superset-of-receiver"),
        pytest.param(Union[int, bool], Union[int, str], id="partially-overlapping-unions-with-primitives"),
        pytest.param(Union[int, Class1], Union[int, Class2], id="partially-overlapping-unions-with-classes"),
        pytest.param(List[int], List[str], id="different-lists-of-primitives"),
        pytest.param(List[int], List, id="list-of-primitive-to-bare-list"),  # is "correct", but we don't support it
        pytest.param(List[int], list, id="list-of-primitive-to-list-object"),  # is "correct", but we don't support it
        pytest.param(List[Class1], List[Class2], id="different-lists-of-classes"),
        pytest.param(List[Class1], List[Class3], id="lists-of-classes-to-subclasses"),
        pytest.param(List[Any], List[str], id="list-of-any-to-list-of-primitives"),
        pytest.param(List[Any], List[Class2], id="list-of-any-to-list-of-classes"),
        pytest.param(
            List[Set[Sequence[str]]], List[Set[Sequence[bool]]], id="nested-sequences-of-different-primitives"
        ),
        pytest.param(
            List[Set[Sequence[str]]], Set[List[Sequence[str]]], id="different-nested-sequences-of-same-primitives"
        ),
        pytest.param(
            List[Set[Sequence[Class1]]], List[Set[Sequence[Class2]]], id="nested-sequences-of-different-classes"
        ),
        pytest.param(
            List[Set[Sequence[Class1]]], List[Set[Sequence[Class3]]], id="nested-sequences-of-classes-to-subclasses"
        ),
        pytest.param(
            List[Set[Sequence[Class1]]], Set[List[Sequence[Class1]]], id="different-nested-sequences-of-same-class"
        ),
        pytest.param(
            List[Set[Sequence[Any]]], List[Set[Sequence[bool]]], id="nested-list-of-Any-to-nested-list-of-primitives"
        ),
        pytest.param(
            List[Set[Sequence[Any]]], List[Set[Sequence[Class2]]], id="nested-list-of-Any-to-nested-list-of-classes"
        ),
        pytest.param(Dict[str, int], Dict[int, int], id="different-dict-of-primitive-keys"),
        pytest.param(Dict[str, int], Dict[str, bool], id="different-dict-of-primitive-values"),
        pytest.param(Dict[str, Class1], Dict[str, Class2], id="different-dict-of-class-values"),
        pytest.param(Dict[str, Class1], Dict[str, Class3], id="different-dict-of-class-to-subclass-values"),
        pytest.param(Dict[Any, int], Dict[int, int], id="dict-of-Any-keys-to-dict-of-primitives"),
        pytest.param(Dict[str, Any], Dict[int, int], id="dict-of-Any-values-to-dict-of-primitives"),
        pytest.param(Dict[str, Any], Dict[int, Class1], id="dict-of-Any-values-to-dict-of-classes"),
        pytest.param(Dict[Any, Any], Dict[int, int], id="dict-of-Any-keys-and-values-to-dict-of-primitives"),
        pytest.param(Dict[Any, Any], Dict[int, Class1], id="dict-of-Any-keys-and-values-to-dict-of-classes"),
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
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[str, Dict[str, Class2]]],
            id="same-nested-mappings-of-different-class-values",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            Dict[str, Mapping[str, Dict[str, Class2]]],
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
            Dict[str, Mapping[str, Dict[str, Class1]]],
            id="nested-mapping-of-Any-values-to-nested-mapping-of-classes",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[Any, Any]]],
            Dict[str, Mapping[str, Dict[str, int]]],
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[str, Dict[Any, Any]]],
            Dict[str, Mapping[str, Dict[str, Class1]]],
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-classes",
        ),
        pytest.param(Literal["a", "b", "c"], Literal["x", "y"], id="different-literal-of-same-primitive"),
        pytest.param(Literal["a", "b", "c"], Literal["a", "b"], id="subset-literal"),
        pytest.param(Literal[Enum1.TEST1], Literal[Enum1.TEST2], id="different-literal-of-same-enum"),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, Class1]]],
            Tuple[Literal["a", "b", "c"], Union[Path, Dict[int, Class1]]],
            id="deeply-nested-complex-type-is-compatible-but-cannot-be-checked",
        ),
    ],
)
def test_connect_non_compatible_types(from_type, to_type):
    comp1 = factory.component_class("Comp1", output_types={"value": from_type})()
    comp2 = factory.component_class("Comp2", input_types={"value": to_type})()

    pipe = Pipeline()
    pipe.add_component("c1", comp1)
    pipe.add_component("c2", comp2)

    with pytest.raises(
        PipelineConnectError,
        match="Cannot connect 'c1.value' with 'c2.value': their declared input and output types do not match.",
    ):
        pipe.connect("c1", "c2")


def test_connect_sender_component_does_not_exist():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("third", "second")


def test_connect_receiver_component_does_not_exist():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(ValueError, match="Component named third not found in the pipeline"):
        pipe.connect("first", "third")


def test_connect_sender_socket_does_not_exist():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="first.wrong does not exist"):
        pipe.connect("first.wrong", "second")


def test_connect_receiver_socket_does_not_exist():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    with pytest.raises(PipelineConnectError, match="second.wrong does not exist"):
        pipe.connect("first", "second.wrong")


def test_connect_many_outputs_to_the_same_input():
    add_1 = AddFixedValue()
    add_2 = AddFixedValue()

    pipe = Pipeline()
    pipe.add_component("first", add_1)
    pipe.add_component("second", add_2)
    pipe.add_component("third", add_2)
    pipe.connect("first.result", "second.value")
    with pytest.raises(PipelineConnectError, match=r"second.value is already connected to \['first'\]"):
        pipe.connect("third.result", "second.value")


def test_connect_many_connections_possible_name_matches():
    Component1 = factory.component_class("Component1", output_types={"value": str})
    Component2 = factory.component_class(
        "Component2", input_types={"value": str, "othervalue": str, "yetanothervalue": str}
    )

    pipe = Pipeline()
    pipe.add_component("c1", Component1())
    pipe.add_component("c2", Component2())
    pipe.connect("c1", "c2")
    assert list(pipe.graph.edges) == [("c1", "c2", "value/value")]


def test_connect_many_connections_possible_no_name_matches():
    Component1 = factory.component_class("Component1", output_types={"value": str})
    Component2 = factory.component_class("Component2", input_types={"value1": str, "value2": str, "value3": str})

    expected_message = re.escape(
        """Cannot connect 'c1' with 'c2': more than one connection is possible between these components. Please specify the connection name, like: pipeline.connect('c1.value', 'c2.value1').
'c1':
 - value: str
'c2':
 - value1: str (available)
 - value2: str (available)
 - value3: str (available)"""
    )

    pipe = Pipeline()
    pipe.add_component("c1", Component1())
    pipe.add_component("c2", Component2())
    with pytest.raises(PipelineConnectError, match=expected_message):
        pipe.connect("c1", "c2")


def test_parse_connection():
    assert parse_connect_string("foobar") == ("foobar", None)
    assert parse_connect_string("foo.bar") == ("foo", "bar")
    assert parse_connect_string("foo.bar.baz") == ("foo", "bar.baz")


def test_connect_with_same_socket_names():
    SimpleComponent = factory.component_class("SimpleComponent", output_types={"documents": List})
    ComponentWithMultipleInputs = factory.component_class(
        "ComponentWithMultipleInputs", input_types={"question": Any, "documents": Any}
    )

    pipe = Pipeline()
    pipe.add_component("simple", SimpleComponent())
    pipe.add_component("multiple", ComponentWithMultipleInputs())

    pipe.connect("simple", "multiple")

    assert list(pipe.graph.edges) == [("simple", "multiple", "documents/documents")]
