# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import pytest

from haystack.dataclasses import ByteStream, ChatMessage, Document, GeneratedAnswer
from haystack.core.component.types import Variadic
from haystack.core.type_utils import _type_name, _types_are_compatible


class Class1:
    pass


class Class2:
    pass


class Class3(Class1):
    pass


class Enum1(Enum):
    TEST1 = Class1
    TEST2 = Class2


@pytest.mark.parametrize(
    "type_,repr_",
    [
        pytest.param(str, "str", id="primitive-types"),
        pytest.param(Any, "Any", id="any"),
        pytest.param(Class1, "Class1", id="class"),
        pytest.param((Optional[int]), "Optional[int]", id="shallow-optional-with-primitive"),
        pytest.param((Optional[Any]), "Optional[Any]", id="shallow-optional-with-any"),
        pytest.param((Optional[Class1]), "Optional[Class1]", id="shallow-optional-with-class"),
        pytest.param((Union[(bool, Class1)]), "Union[bool, Class1]", id="shallow-union"),
        pytest.param((List[str]), "List[str]", id="shallow-sequence-of-primitives"),
        pytest.param((List[Set[Sequence[str]]]), "List[Set[Sequence[str]]]", id="nested-sequence-of-primitives"),
        pytest.param(
            (Optional[List[Set[Sequence[str]]]]),
            "Optional[List[Set[Sequence[str]]]]",
            id="optional-nested-sequence-of-primitives",
        ),
        pytest.param(
            (List[Set[Sequence[Optional[str]]]]),
            "List[Set[Sequence[Optional[str]]]]",
            id="nested-optional-sequence-of-primitives",
        ),
        pytest.param((List[Class1]), "List[Class1]", id="shallow-sequence-of-classes"),
        pytest.param((List[Set[Sequence[Class1]]]), "List[Set[Sequence[Class1]]]", id="nested-sequence-of-classes"),
        pytest.param((Dict[(str, int)]), "Dict[str, int]", id="shallow-mapping-of-primitives"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            "Dict[str, Mapping[str, Dict[str, int]]]",
            id="nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(Any, Dict[(str, int)])])]),
            "Dict[str, Mapping[Any, Dict[str, int]]]",
            id="nested-mapping-of-primitives-with-any",
        ),
        pytest.param((Dict[(str, Class1)]), "Dict[str, Class1]", id="shallow-mapping-of-classes"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            "Dict[str, Mapping[str, Dict[str, Class1]]]",
            id="nested-mapping-of-classes",
        ),
        pytest.param((Literal["a", "b", "c"]), "Literal['a', 'b', 'c']", id="string-literal"),
        pytest.param((Literal[1, 2, 3]), "Literal[1, 2, 3]", id="primitive-literal"),
        pytest.param((Literal[Enum1.TEST1]), "Literal[Enum1.TEST1]", id="enum-literal"),
        pytest.param(
            (Tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])])]),
            "Tuple[Optional[Literal['a', 'b', 'c']], Union[Path, Dict[int, Class1]]]",
            id="deeply-nested-complex-type",
        ),
    ],
)
def test_type_name(type_, repr_):
    assert _type_name(type_) == repr_


# Basic type groups
primitive_types = [bool, bytes, dict, float, int, list, str, Path]
class_types = [Class1, Class3]
container_types = [
    Iterable[int],
    List[int],
    List[Class1],
    Set[str],
    Dict[str, int],
    Dict[str, Class1],
    Sequence[bool],
    Mapping[str, Dict[str, int]],
    Variadic[int],
]
nested_container_types = [
    List[Set[Sequence[bool]]],
    List[Set[Sequence[Class1]]],
    Dict[str, Mapping[str, Dict[str, int]]],
    Dict[str, Mapping[str, Dict[str, Class1]]],
    Tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])])],
]
literals = [Literal["a", "b", "c"], Literal[Enum1.TEST1]]
haystack_types = [ByteStream, ChatMessage, Document, GeneratedAnswer]
extras = [Union[int, str]]


def generate_id(sender, receiver):
    """Helper function to generate readable IDs"""
    sender_name = _type_name(sender)
    receiver_name = _type_name(receiver)
    return f"{sender_name}-to-{receiver_name}"


def generate_symmetric_test_cases():
    test_cases = []
    # Identity: same type should be compatible
    for t in (
        primitive_types + class_types + container_types + literals + haystack_types + nested_container_types + extras
    ):
        test_cases.append(pytest.param(t, t, id=generate_id(t, t)))
    return test_cases


@pytest.mark.parametrize("sender_type, receiver_type", generate_symmetric_test_cases())
def test_same_types_are_compatible(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)


# TODO Add tests for
#  - Document subclasses
def generate_asymmetric_test_cases():
    test_cases = []

    # Primitives: Optional, Union, Any
    for t in (
        primitive_types + class_types + container_types + literals + haystack_types + nested_container_types + extras
    ):
        test_cases.append(pytest.param(t, Optional[t], id=generate_id(t, Optional[t])))
        test_cases.append(pytest.param(t, Union[t, complex], id=generate_id(t, Union[t, complex])))
        test_cases.append(pytest.param(t, Any, id=generate_id(t, Any)))

    # Classes: Optional, Union, Any
    for cls in class_types:
        test_cases.append(pytest.param(cls, Optional[cls], id=generate_id(cls, Optional[cls])))
        test_cases.append(pytest.param(cls, Union[cls, complex], id=generate_id(cls, Union[cls, complex])))
        test_cases.append(pytest.param(cls, Any, id=generate_id(cls, Any)))

    # Subclasses:
    # Subclass → Superclass
    test_cases.append(pytest.param(Class3, Class1, id=generate_id(Class3, Class1)))
    # Subclass → Union of Superclass and other type
    test_cases.append(pytest.param(Class3, Union[int, Class1], id=generate_id(Class3, Union[int, Class1])))
    # List of Class3 → List of Class1
    test_cases.append(pytest.param(List[Class3], List[Class1], id=generate_id(List[Class3], List[Class1])))
    # Dict of subclass → Dict of superclass
    test_cases.append(
        pytest.param(Dict[str, Class3], Dict[str, Class1], id=generate_id(Dict[str, Class3], Dict[str, Class1]))
    )

    # Containers: Optional, Union, and Any compatibility
    for container in container_types:
        test_cases.append(pytest.param(container, Optional[container], id=generate_id(container, Optional[container])))
        test_cases.append(
            pytest.param(container, Union[container, int], id=generate_id(container, Union[container, int]))
        )
        test_cases.append(pytest.param(container, Any, id=generate_id(container, Any)))

    return test_cases


@pytest.mark.parametrize("receiver_type, sender_type", generate_asymmetric_test_cases())
def test_asymmetric_types_are_compatible(receiver_type, sender_type):
    assert _types_are_compatible(sender_type, receiver_type)


@pytest.mark.parametrize("receiver_type, sender_type", generate_asymmetric_test_cases())
def test_asymmetric_types_are_not_compatible_strict(receiver_type, sender_type):
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize("receiver_type, sender_type", generate_asymmetric_test_cases())
def test_asymmetric_types_are_compatible_relaxed(receiver_type, sender_type):
    assert _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param((List[int]), (List[Any]), id="list-of-primitive-to-list-of-any"),
        pytest.param((List[Class1]), (List[Any]), id="list-of-classes-to-list-of-any"),
        pytest.param(
            (List[Set[Sequence[bool]]]),
            (List[Set[Sequence[Any]]]),
            id="nested-sequences-of-primitives-to-nested-sequences-of-any",
        ),
        pytest.param(
            (List[Set[Sequence[Class3]]]),
            (List[Set[Sequence[Class1]]]),
            id="nested-sequences-of-subclasses-to-nested-sequences-of-classes",
        ),
        pytest.param(
            (List[Set[Sequence[Class1]]]),
            (List[Set[Sequence[Any]]]),
            id="nested-sequences-of-classes-to-nested-sequences-of-any",
        ),
        pytest.param((Dict[(str, int)]), (Dict[(Any, int)]), id="dict-of-primitives-to-dict-of-any-keys"),
        pytest.param((Dict[(str, int)]), (Dict[(str, Any)]), id="dict-of-primitives-to-dict-of-any-values"),
        pytest.param((Dict[(str, int)]), (Dict[(Any, Any)]), id="dict-of-primitives-to-dict-of-any-key-and-values"),
        pytest.param((Dict[(str, Class3)]), (Dict[(str, Class1)]), id="dict-of-subclasses-to-dict-of-classes"),
        pytest.param((Dict[(str, Class1)]), (Dict[(Any, Class1)]), id="dict-of-classes-to-dict-of-any-keys"),
        pytest.param((Dict[(str, Class1)]), (Dict[(str, Any)]), id="dict-of-classes-to-dict-of-any-values"),
        pytest.param((Dict[(str, Class1)]), (Dict[(Any, Any)]), id="dict-of-classes-to-dict-of-any-key-and-values"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            id="nested-mappings-of-same-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(Any, int)])])]),
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(Any, Dict[(str, int)])])]),
            id="nested-mapping-of-primitives-to-nested-mapping-of-higher-level-any-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Any)])])]),
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-values",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(Any, Dict[(Any, Any)])])]),
            id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys-and-values",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            id="nested-mappings-of-same-classes",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class3)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            id="nested-mapping-of-subclasses-to-nested-mapping-of-classes",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(str, Dict[(Any, Class1)])])]),
            id="nested-mapping-of-classes-to-nested-mapping-of-any-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(Any, Dict[(str, Class1)])])]),
            id="nested-mapping-of-classes-to-nested-mapping-of-higher-level-any-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Any)])])]),
            id="nested-mapping-of-classes-to-nested-mapping-of-any-values",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(Any, Dict[(Any, Any)])])]),
            id="nested-mapping-of-classes-to-nested-mapping-of-any-keys-and-values",
        ),
    ],
)
def test_asymmetric_container_types_are_compatible(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)
    # TODO Reverse tests and add strict and relaxed tests


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(int, bool, id="different-primitives"),
        pytest.param(Class1, Class3, id="class-to-subclass"),
        pytest.param((Optional[str]), str, id="sending-primitive-is-optional"),
        pytest.param((Optional[Class1]), Class1, id="sending-class-is-optional"),
        pytest.param((Optional[List[int]]), (List[int]), id="sending-list-is-optional"),
        pytest.param((Union[(int, str)]), str, id="sending-type-is-union"),
        pytest.param((Union[(int, str, bool)]), (Union[(int, str)]), id="sending-union-is-superset-of-receiver"),
        pytest.param((Union[(int, bool)]), (Union[(int, str)]), id="partially-overlapping-unions-with-primitives"),
        pytest.param((Union[(int, Class1)]), (Union[(int, Class2)]), id="partially-overlapping-unions-with-classes"),
        pytest.param((List[int]), List, id="list-of-primitive-to-bare-list"),
        pytest.param((List[int]), list, id="list-of-primitive-to-list-object"),
        pytest.param((List[Class1]), (List[Class3]), id="lists-of-classes-to-subclasses"),
        pytest.param((List[Any]), (List[str]), id="list-of-any-to-list-of-primitives"),
        pytest.param((List[Any]), (List[Class2]), id="list-of-any-to-list-of-classes"),
        pytest.param(
            (List[Set[Sequence[Class1]]]), (List[Set[Sequence[Class3]]]), id="nested-sequences-of-classes-to-subclasses"
        ),
        pytest.param(
            (List[Set[Sequence[Any]]]),
            (List[Set[Sequence[bool]]]),
            id="nested-list-of-Any-to-nested-list-of-primitives",
        ),
        pytest.param((Dict[(str, Class1)]), (Dict[(str, Class3)]), id="different-dict-of-class-to-subclass-values"),
        pytest.param((Dict[(Any, int)]), (Dict[(int, int)]), id="dict-of-Any-keys-to-dict-of-primitives"),
        pytest.param((Dict[(Any, Any)]), (Dict[(int, int)]), id="dict-of-Any-keys-and-values-to-dict-of-primitives"),
        pytest.param((Dict[(Any, Any)]), (Dict[(int, Class1)]), id="dict-of-Any-keys-and-values-to-dict-of-classes"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(Any, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            id="nested-mapping-of-Any-keys-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(Any, Dict[(Any, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            id="nested-mapping-of-higher-level-Any-keys-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Any)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            id="nested-mapping-of-Any-values-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Any)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            id="nested-mapping-of-Any-values-to-nested-mapping-of-classes",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(Any, Any)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(Any, Any)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            id="nested-mapping-of-Any-keys-and-values-to-nested-mapping-of-classes",
        ),
        pytest.param((Literal["a", "b", "c"]), (Literal["a", "b"]), id="subset-literal"),
        pytest.param(
            (Tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])])]),
            (Tuple[(Literal["a", "b", "c"], Union[(Path, Dict[(int, Class1)])])]),
            id="deeply-nested-complex-type-is-compatible-but-cannot-be-checked",
        ),
        pytest.param(
            (List[Set[Sequence[Any]]]), (List[Set[Sequence[Class2]]]), id="nested-list-of-Any-to-nested-list-of-classes"
        ),
    ],
)
def test_types_are_not_compatible_strict(sender_type, receiver_type):
    assert not _types_are_compatible(sender_type, receiver_type)
    # TODO Add relaxed version of test


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(int, str, id="different-primitives"),
        pytest.param(Class1, Class2, id="different-classes"),
        pytest.param((List[int]), (List[str]), id="different-lists-of-primitives"),
        pytest.param((List[Class1]), (List[Class2]), id="different-lists-of-classes"),
        pytest.param(
            (List[Set[Sequence[str]]]), (List[Set[Sequence[bool]]]), id="nested-sequences-of-different-primitives"
        ),
        pytest.param(
            (List[Set[Sequence[str]]]), (Set[List[Sequence[str]]]), id="different-nested-sequences-of-same-primitives"
        ),
        pytest.param(
            (List[Set[Sequence[Class1]]]), (List[Set[Sequence[Class2]]]), id="nested-sequences-of-different-classes"
        ),
        pytest.param(
            (List[Set[Sequence[Class1]]]), (Set[List[Sequence[Class1]]]), id="different-nested-sequences-of-same-class"
        ),
        pytest.param((Dict[(str, int)]), (Dict[(int, int)]), id="different-dict-of-primitive-keys"),
        pytest.param((Dict[(str, int)]), (Dict[(str, float)]), id="different-dict-of-primitive-values"),
        pytest.param((Dict[(str, Class1)]), (Dict[(str, Class2)]), id="different-dict-of-class-values"),
        pytest.param((Dict[(str, Any)]), (Dict[(int, int)]), id="dict-of-Any-values-to-dict-of-primitives"),
        pytest.param((Dict[(str, Any)]), (Dict[(int, Class1)]), id="dict-of-Any-values-to-dict-of-classes"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Mapping[(str, Dict[(str, Dict[(str, int)])])]),
            id="different-nested-mappings-of-same-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(int, int)])])]),
            id="same-nested-mappings-of-different-primitive-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(int, Dict[(str, int)])])]),
            id="same-nested-mappings-of-different-higher-level-primitive-keys",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, dict)])])]),
            id="same-nested-mappings-of-different-primitive-values",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class2)])])]),
            id="same-nested-mappings-of-different-class-values",
        ),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            (Dict[(str, Mapping[(str, Dict[(str, Class2)])])]),
            id="same-nested-mappings-of-class-to-subclass-values",
        ),
        pytest.param((Literal["a", "b", "c"]), (Literal["x", "y"]), id="different-literal-of-same-primitive"),
        pytest.param((Literal[Enum1.TEST1]), (Literal[Enum1.TEST2]), id="different-literal-of-same-enum"),
    ],
)
def test_types_are_always_not_compatible(sender_type, receiver_type):
    assert not _types_are_compatible(sender_type, receiver_type)
