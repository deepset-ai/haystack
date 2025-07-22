# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import pytest

from haystack.core.component.types import Variadic
from haystack.core.type_utils import _type_name, _types_are_compatible
from haystack.dataclasses import ByteStream, ChatMessage, Document, GeneratedAnswer


class Class1:
    pass


class Class2:
    pass


class Class3(Class1):
    pass


class Enum1(Enum):
    TEST1 = Class1
    TEST2 = Class2


# Basic type groups
primitive_types = [bool, bytes, dict, float, int, list, str, Path]
class_types = [Class1, Class3]
container_types = [
    # haystack
    Variadic[int],
    # typing
    Dict[str, int],
    Dict[str, Class1],
    Iterable[int],
    List[int],
    List[Class1],
    Mapping[str, Dict[str, int]],
    Set[str],
    Sequence[bool],
    Tuple[int],
    # builtins
    dict[str, int],
    dict[str, Class1],
    list[int],
    list[Class1],
    set[str],
    tuple[int],
]
nested_container_types = [
    # typing
    List[Set[Sequence[bool]]],
    List[Set[Sequence[Class1]]],
    Dict[str, Mapping[str, Dict[str, int]]],
    Dict[str, Mapping[str, Dict[str, Class1]]],
    Tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])])],
    # builtins
    list[set[Sequence[bool]]],
    list[set[Sequence[Class1]]],
    dict[str, Mapping[str, dict[str, int]]],
    dict[str, Mapping[str, dict[str, Class1]]],
    tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, dict[(int, Class1)])])],
]
literals = [Literal["a", "b", "c"], Literal[Enum1.TEST1]]
haystack_types = [ByteStream, ChatMessage, Document, GeneratedAnswer]
extras = [Union[int, str]]


def generate_id(sender, receiver):
    """Generate readable test case IDs."""
    return f"{_type_name(sender)}-to-{_type_name(receiver)}"


def generate_symmetric_cases():
    """Generate symmetric test cases where sender and receiver types are the same."""
    return [
        pytest.param(t, t, id=generate_id(t, t))
        for t in (
            primitive_types
            + class_types
            + container_types
            + literals
            + haystack_types
            + nested_container_types
            + extras
        )
    ]


def generate_strict_asymmetric_cases():
    """Generate asymmetric test cases with different sender and receiver types."""
    cases = []

    # Primitives: Optional, Union, Any
    for t in (
        primitive_types + class_types + container_types + literals + haystack_types + nested_container_types + extras
    ):
        cases.append(pytest.param(t, Optional[t], id=generate_id(t, Optional[t])))
        cases.append(pytest.param(t, Union[t, complex], id=generate_id(t, Union[t, complex])))
        cases.append(pytest.param(t, Any, id=generate_id(t, Any)))

    # Classes: Optional, Union, Any
    for cls in class_types:
        cases.append(pytest.param(cls, Optional[cls], id=generate_id(cls, Optional[cls])))
        cases.append(pytest.param(cls, Union[cls, complex], id=generate_id(cls, Union[cls, complex])))
        cases.append(pytest.param(cls, Any, id=generate_id(cls, Any)))

    # Subclass → Superclass
    cases.extend(
        [
            # Subclass → Superclass
            pytest.param(Class3, Class1, id=generate_id(Class3, Class1)),
            # Subclass → Union of Superclass and other type
            pytest.param(Class3, Union[int, Class1], id=generate_id(Class3, Union[int, Class1])),
            # List of Class3 → List of Class1
            pytest.param(List[Class3], List[Class1], id=generate_id(List[Class3], List[Class1])),
            pytest.param(list[Class3], list[Class1], id=generate_id(list[Class3], list[Class1])),
            pytest.param(list[Class3], List[Class1], id=generate_id(list[Class3], List[Class1])),
            # Dict of subclass → Dict of superclass
            pytest.param(Dict[str, Class3], Dict[str, Class1], id=generate_id(Dict[str, Class3], Dict[str, Class1])),
            pytest.param(dict[str, Class3], dict[str, Class1], id=generate_id(dict[str, Class3], dict[str, Class1])),
            pytest.param(dict[str, Class3], Dict[str, Class1], id=generate_id(dict[str, Class3], Dict[str, Class1])),
        ]
    )

    # Containers: Optional, Union, and Any compatibility
    for container in container_types:
        cases.append(pytest.param(container, Optional[container], id=generate_id(container, Optional[container])))
        cases.append(pytest.param(container, Union[container, int], id=generate_id(container, Union[container, int])))
        cases.append(pytest.param(container, Any, id=generate_id(container, Any)))

    # Extra cases
    cases.extend(
        [
            pytest.param(bool, int, id="different-primitives"),
            pytest.param((Literal["a", "b"]), (Literal["a", "b", "c"]), id="sending-subset-literal"),
        ]
    )

    # typing Extra container cases
    cases.extend(
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
            pytest.param(
                (Tuple[Literal["a", "b", "c"], Union[(Path, Dict[(int, Class1)])]]),
                (Tuple[Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])]]),
                id="deeply-nested-complex-type",
            ),
        ]
    )

    # builtins Extra container cases
    cases.extend(
        [
            pytest.param((list[int]), (list[Any]), id="list-of-primitive-to-list-of-any"),
            pytest.param((list[Class1]), (list[Any]), id="list-of-classes-to-list-of-any"),
            pytest.param(
                (list[set[Sequence[bool]]]),
                (list[set[Sequence[Any]]]),
                id="nested-sequences-of-primitives-to-nested-sequences-of-any",
            ),
            pytest.param(
                (list[set[Sequence[Class3]]]),
                (list[set[Sequence[Class1]]]),
                id="nested-sequences-of-subclasses-to-nested-sequences-of-classes",
            ),
            pytest.param(
                (list[set[Sequence[Class1]]]),
                (list[set[Sequence[Any]]]),
                id="nested-sequences-of-classes-to-nested-sequences-of-any",
            ),
            pytest.param((dict[(str, int)]), (dict[(Any, int)]), id="dict-of-primitives-to-dict-of-any-keys"),
            pytest.param((dict[(str, int)]), (dict[(str, Any)]), id="dict-of-primitives-to-dict-of-any-values"),
            pytest.param((dict[(str, int)]), (dict[(Any, Any)]), id="dict-of-primitives-to-dict-of-any-key-and-values"),
            pytest.param((dict[(str, Class3)]), (dict[(str, Class1)]), id="dict-of-subclasses-to-dict-of-classes"),
            pytest.param((dict[(str, Class1)]), (dict[(Any, Class1)]), id="dict-of-classes-to-dict-of-any-keys"),
            pytest.param((dict[(str, Class1)]), (dict[(str, Any)]), id="dict-of-classes-to-dict-of-any-values"),
            pytest.param((dict[(str, Class1)]), (dict[(Any, Any)]), id="dict-of-classes-to-dict-of-any-key-and-values"),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, int)])])]),
                (dict[(str, Mapping[(str, dict[(Any, int)])])]),
                id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, int)])])]),
                (dict[(str, Mapping[(Any, dict[(str, int)])])]),
                id="nested-mapping-of-primitives-to-nested-mapping-of-higher-level-any-keys",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, int)])])]),
                (dict[(str, Mapping[(str, dict[(str, Any)])])]),
                id="nested-mapping-of-primitives-to-nested-mapping-of-any-values",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, int)])])]),
                (dict[(str, Mapping[(Any, dict[(Any, Any)])])]),
                id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys-and-values",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, Class3)])])]),
                (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
                id="nested-mapping-of-subclasses-to-nested-mapping-of-classes",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
                (dict[(str, Mapping[(str, dict[(Any, Class1)])])]),
                id="nested-mapping-of-classes-to-nested-mapping-of-any-keys",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
                (dict[(str, Mapping[(Any, dict[(str, Class1)])])]),
                id="nested-mapping-of-classes-to-nested-mapping-of-higher-level-any-keys",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
                (dict[(str, Mapping[(str, dict[(str, Any)])])]),
                id="nested-mapping-of-classes-to-nested-mapping-of-any-values",
            ),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
                (dict[(str, Mapping[(Any, dict[(Any, Any)])])]),
                id="nested-mapping-of-classes-to-nested-mapping-of-any-keys-and-values",
            ),
            pytest.param(
                (tuple[Literal["a", "b", "c"], Union[(Path, dict[(int, Class1)])]]),
                (tuple[Optional[Literal["a", "b", "c"]], Union[(Path, dict[(int, Class1)])]]),
                id="deeply-nested-complex-type",
            ),
        ]
    )

    # mixing typing and builtins extra container cases
    cases.extend(
        [
            pytest.param((list[int]), (List[Any]), id="list-of-primitive-to-list-of-any"),
            pytest.param((list[Class1]), (List[Any]), id="list-of-classes-to-list-of-any"),
            pytest.param(
                (list[set[Sequence[bool]]]),
                (List[Set[Sequence[Any]]]),
                id="nested-sequences-of-primitives-to-nested-sequences-of-any",
            ),
            pytest.param((dict[(str, int)]), (Dict[(Any, int)]), id="dict-of-primitives-to-dict-of-any-keys"),
            pytest.param(
                (dict[(str, Mapping[(str, dict[(str, int)])])]),
                (Dict[(str, Mapping[(str, Dict[(Any, int)])])]),
                id="nested-mapping-of-primitives-to-nested-mapping-of-any-keys",
            ),
            pytest.param(
                (tuple[Literal["a", "b", "c"], Union[(Path, dict[(int, Class1)])]]),
                (Tuple[Optional[Literal["a", "b", "c"]], Union[(Path, Dict[(int, Class1)])]]),
                id="deeply-nested-complex-type",
            ),
        ]
    )

    return cases


# Precompute test cases for reuse
symmetric_cases = generate_symmetric_cases()
asymmetric_cases = generate_strict_asymmetric_cases()


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
        pytest.param((list[str]), "list[str]", id="shallow-sequence-of-primitives"),
        pytest.param((List[Set[Sequence[str]]]), "List[Set[Sequence[str]]]", id="nested-sequence-of-primitives"),
        pytest.param((list[set[Sequence[str]]]), "list[set[Sequence[str]]]", id="nested-sequence-of-primitives"),
        pytest.param(
            (Optional[List[Set[Sequence[str]]]]),
            "Optional[List[Set[Sequence[str]]]]",
            id="optional-nested-sequence-of-primitives",
        ),
        pytest.param(
            (Optional[list[set[Sequence[str]]]]),
            "Optional[list[set[Sequence[str]]]]",
            id="optional-nested-sequence-of-primitives",
        ),
        pytest.param(
            (List[Set[Sequence[Optional[str]]]]),
            "List[Set[Sequence[Optional[str]]]]",
            id="nested-optional-sequence-of-primitives",
        ),
        pytest.param(
            (list[set[Sequence[Optional[str]]]]),
            "list[set[Sequence[Optional[str]]]]",
            id="nested-optional-sequence-of-primitives",
        ),
        pytest.param((List[Class1]), "List[Class1]", id="shallow-sequence-of-classes"),
        pytest.param((list[Class1]), "list[Class1]", id="shallow-sequence-of-classes"),
        pytest.param((List[Set[Sequence[Class1]]]), "List[Set[Sequence[Class1]]]", id="nested-sequence-of-classes"),
        pytest.param((list[set[Sequence[Class1]]]), "list[set[Sequence[Class1]]]", id="nested-sequence-of-classes"),
        pytest.param((Dict[(str, int)]), "Dict[str, int]", id="shallow-mapping-of-primitives"),
        pytest.param((dict[(str, int)]), "dict[str, int]", id="shallow-mapping-of-primitives"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
            "Dict[str, Mapping[str, Dict[str, int]]]",
            id="nested-mapping-of-primitives",
        ),
        pytest.param(
            (dict[(str, Mapping[(str, dict[(str, int)])])]),
            "dict[str, Mapping[str, dict[str, int]]]",
            id="nested-mapping-of-primitives",
        ),
        pytest.param(
            (Dict[(str, Mapping[(Any, Dict[(str, int)])])]),
            "Dict[str, Mapping[Any, Dict[str, int]]]",
            id="nested-mapping-of-primitives-with-any",
        ),
        pytest.param(
            (dict[(str, Mapping[(Any, dict[(str, int)])])]),
            "dict[str, Mapping[Any, dict[str, int]]]",
            id="nested-mapping-of-primitives-with-any",
        ),
        pytest.param((Dict[(str, Class1)]), "Dict[str, Class1]", id="shallow-mapping-of-classes"),
        pytest.param((dict[(str, Class1)]), "dict[str, Class1]", id="shallow-mapping-of-classes"),
        pytest.param(
            (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
            "Dict[str, Mapping[str, Dict[str, Class1]]]",
            id="nested-mapping-of-classes",
        ),
        pytest.param(
            (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
            "dict[str, Mapping[str, dict[str, Class1]]]",
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
        pytest.param(
            (tuple[(Optional[Literal["a", "b", "c"]], Union[(Path, dict[(int, Class1)])])]),
            "tuple[Optional[Literal['a', 'b', 'c']], Union[Path, dict[int, Class1]]]",
            id="deeply-nested-complex-type",
        ),
    ],
)
def test_type_name(type_, repr_):
    assert _type_name(type_) == repr_


@pytest.mark.parametrize("sender_type, receiver_type", symmetric_cases)
def test_same_types_are_compatible_strict(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type, True)


@pytest.mark.parametrize("sender_type, receiver_type", asymmetric_cases)
def test_asymmetric_types_are_compatible_strict(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type, True)


@pytest.mark.parametrize("sender_type, receiver_type", asymmetric_cases)
def test_asymmetric_types_are_not_compatible_strict(sender_type, receiver_type):
    assert not _types_are_compatible(receiver_type, sender_type, True)


incompatible_type_cases = [
    pytest.param(Tuple[int, str], Tuple[Any], id="tuple-of-primitive-to-tuple-of-any-different-lengths"),
    pytest.param(tuple[int, str], tuple[Any], id="tuple-of-primitive-to-tuple-of-any-different-lengths"),
    pytest.param(int, str, id="different-primitives"),
    pytest.param(Class1, Class2, id="different-classes"),
    pytest.param((List[int]), (List[str]), id="different-lists-of-primitives"),
    pytest.param((list[int]), (list[str]), id="different-lists-of-primitives"),
    pytest.param((List[Class1]), (List[Class2]), id="different-lists-of-classes"),
    pytest.param((list[Class1]), (list[Class2]), id="different-lists-of-classes"),
    pytest.param((Literal["a", "b", "c"]), (Literal["x", "y"]), id="different-literal-of-same-primitive"),
    pytest.param((Literal[Enum1.TEST1]), (Literal[Enum1.TEST2]), id="different-literal-of-same-enum"),
    pytest.param(
        (List[Set[Sequence[str]]]), (List[Set[Sequence[bool]]]), id="nested-sequences-of-different-primitives"
    ),
    pytest.param(
        (list[set[Sequence[str]]]), (list[set[Sequence[bool]]]), id="nested-sequences-of-different-primitives"
    ),
    pytest.param(
        (List[Set[Sequence[str]]]), (Set[List[Sequence[str]]]), id="different-nested-sequences-of-same-primitives"
    ),
    pytest.param(
        (list[set[Sequence[str]]]), (set[list[Sequence[str]]]), id="different-nested-sequences-of-same-primitives"
    ),
    pytest.param(
        (List[Set[Sequence[Class1]]]), (List[Set[Sequence[Class2]]]), id="nested-sequences-of-different-classes"
    ),
    pytest.param(
        (list[set[Sequence[Class1]]]), (list[set[Sequence[Class2]]]), id="nested-sequences-of-different-classes"
    ),
    pytest.param(
        (List[Set[Sequence[Class1]]]), (Set[List[Sequence[Class1]]]), id="different-nested-sequences-of-same-class"
    ),
    pytest.param(
        (list[set[Sequence[Class1]]]), (set[list[Sequence[Class1]]]), id="different-nested-sequences-of-same-class"
    ),
    pytest.param((Dict[(str, int)]), (Dict[(int, int)]), id="different-dict-of-primitive-keys"),
    pytest.param((dict[(str, int)]), (dict[(int, int)]), id="different-dict-of-primitive-keys"),
    pytest.param((Dict[(str, int)]), (Dict[(str, float)]), id="different-dict-of-primitive-values"),
    pytest.param((dict[(str, int)]), (dict[(str, float)]), id="different-dict-of-primitive-values"),
    pytest.param((Dict[(str, Class1)]), (Dict[(str, Class2)]), id="different-dict-of-class-values"),
    pytest.param((dict[(str, Class1)]), (dict[(str, Class2)]), id="different-dict-of-class-values"),
    pytest.param((Dict[(str, Any)]), (Dict[(int, int)]), id="dict-of-Any-values-to-dict-of-primitives"),
    pytest.param((dict[(str, Any)]), (dict[(int, int)]), id="dict-of-Any-values-to-dict-of-primitives"),
    pytest.param((Dict[(str, Any)]), (Dict[(int, Class1)]), id="dict-of-Any-values-to-dict-of-classes"),
    pytest.param((dict[(str, Any)]), (dict[(int, Class1)]), id="dict-of-Any-values-to-dict-of-classes"),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
        (Mapping[(str, Dict[(str, Dict[(str, int)])])]),
        id="different-nested-mappings-of-same-primitives",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, int)])])]),
        (Mapping[(str, dict[(str, dict[(str, int)])])]),
        id="different-nested-mappings-of-same-primitives",
    ),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
        (Dict[(str, Mapping[(str, Dict[(int, int)])])]),
        id="same-nested-mappings-of-different-primitive-keys",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, int)])])]),
        (dict[(str, Mapping[(str, dict[(int, int)])])]),
        id="same-nested-mappings-of-different-primitive-keys",
    ),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
        (Dict[(str, Mapping[(int, Dict[(str, int)])])]),
        id="same-nested-mappings-of-different-higher-level-primitive-keys",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, int)])])]),
        (dict[(str, Mapping[(int, dict[(str, int)])])]),
        id="same-nested-mappings-of-different-higher-level-primitive-keys",
    ),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, int)])])]),
        (Dict[(str, Mapping[(str, Dict[(str, dict)])])]),
        id="same-nested-mappings-of-different-primitive-values",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, int)])])]),
        (dict[(str, Mapping[(str, dict[(str, dict)])])]),
        id="same-nested-mappings-of-different-primitive-values",
    ),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
        (Dict[(str, Mapping[(str, Dict[(str, Class2)])])]),
        id="same-nested-mappings-of-different-class-values",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
        (dict[(str, Mapping[(str, dict[(str, Class2)])])]),
        id="same-nested-mappings-of-different-class-values",
    ),
    pytest.param(
        (Dict[(str, Mapping[(str, Dict[(str, Class1)])])]),
        (Dict[(str, Mapping[(str, Dict[(str, Class2)])])]),
        id="same-nested-mappings-of-class-to-subclass-values",
    ),
    pytest.param(
        (dict[(str, Mapping[(str, dict[(str, Class1)])])]),
        (dict[(str, Mapping[(str, dict[(str, Class2)])])]),
        id="same-nested-mappings-of-class-to-subclass-values",
    ),
]


@pytest.mark.parametrize("sender_type, receiver_type", incompatible_type_cases)
def test_types_are_always_not_compatible_strict(sender_type, receiver_type):
    assert not _types_are_compatible(sender_type, receiver_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param((Union[(int, bool)]), (Union[(int, str)]), id="partially-overlapping-unions-with-primitives"),
        pytest.param((Union[(int, Class1)]), (Union[(int, Class2)]), id="partially-overlapping-unions-with-classes"),
    ],
)
def test_partially_overlapping_unions_are_not_compatible_strict(sender_type, receiver_type):
    assert not _types_are_compatible(sender_type, receiver_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(List[int], List, id="list-of-primitive-to-bare-list"),
        pytest.param(Dict[str, int], Dict, id="dict-of-primitive-to-bare-dict"),
        pytest.param(Set[float], Set, id="set-of-primitive-to-bare-set"),
        pytest.param(Tuple[int, str], Tuple, id="tuple-of-primitive-to-bare-tuple"),
        # TODO These tests are failing
        pytest.param(list[int], list, id="list-of-primitive-to-bare-list"),
        pytest.param(dict[str, int], dict, id="dict-of-primitive-to-bare-dict"),
        pytest.param(set[float], set, id="set-of-primitive-to-bare-set"),
        pytest.param(tuple[int, str], tuple, id="tuple-of-primitive-to-bare-tuple"),
    ],
)
def test_container_of_primitive_to_bare_container_strict(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)
    # Bare container types should not be compatible with their typed counterparts
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(List[Any], List, id="list-of-any-to-bare-list"),
        pytest.param(Dict[Any, Any], Dict, id="dict-of-any-to-bare-dict"),
        pytest.param(Set[Any], Set, id="set-of-any-to-bare-set"),
        pytest.param(Tuple[Any], Tuple, id="tuple-of-any-to-bare-tuple"),
        pytest.param(Callable[[Any], Any], Callable, id="callable-of-any-to-bare-callable"),
        pytest.param(list[Any], list, id="list-of-any-to-bare-list"),
        pytest.param(dict[Any, Any], dict, id="dict-of-any-to-bare-dict"),
        pytest.param(set[Any], set, id="set-of-any-to-bare-set"),
        pytest.param(tuple[Any], tuple, id="tuple-of-any-to-bare-tuple"),
    ],
)
def test_container_of_any_to_bare_container_strict(sender_type, receiver_type):
    # Both are compatible
    assert _types_are_compatible(sender_type, receiver_type)
    assert _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(Literal["test"], Literal, id="literal-to-bare-literal"),
        pytest.param(Union[str, int], Union, id="union-to-bare-union"),
        pytest.param(Optional[str], Optional, id="optional-to-bare-optional"),
    ],
)
def test_always_incompatible_bare_types(sender_type, receiver_type):
    # Neither are compatible
    assert not _types_are_compatible(sender_type, receiver_type)
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        # Nested Lists
        pytest.param(List[List[int]], List[List], id="nested-list-to-partially-bare-list"),
        pytest.param(List[List[int]], List, id="nested-list-to-bare-list"),
        pytest.param(list[list[int]], list[list], id="nested-list-to-partially-bare-list"),
        pytest.param(list[list[int]], list, id="nested-list-to-bare-list"),
        # Nested Dicts
        pytest.param(Dict[str, Dict[str, int]], Dict[str, Dict], id="nested-dict-to-partially-bare-dict"),
        pytest.param(Dict[str, Dict[str, int]], Dict, id="nested-dict-to-bare-dict"),
        pytest.param(dict[str, dict[str, int]], dict[str, dict], id="nested-dict-to-partially-bare-dict"),
        pytest.param(dict[str, dict[str, int]], dict, id="nested-dict-to-bare-dict"),
        # Nested Sets
        pytest.param(Set[Set[int]], Set[Set], id="nested-set-to-partially-bare-set"),
        pytest.param(Set[Set[int]], Set, id="nested-set-to-bare-set"),
        pytest.param(set[set[int]], set[set], id="nested-set-to-partially-bare-set"),
        pytest.param(set[set[int]], set, id="nested-set-to-bare-set"),
        # Nested Tuples
        pytest.param(Tuple[Tuple[int, str], bool], Tuple[Tuple, bool], id="nested-tuple-to-partially-bare-tuple"),
        pytest.param(Tuple[Tuple[int, str], bool], Tuple, id="nested-tuple-to-bare-tuple"),
        pytest.param(tuple[tuple[int, str], bool], tuple[tuple, bool], id="nested-tuple-to-partially-bare-tuple"),
        pytest.param(tuple[tuple[int, str], bool], tuple, id="nested-tuple-to-bare-tuple"),
        # Mixed nesting
        pytest.param(List[Dict[str, int]], List[Dict], id="list-of-dict-to-list-of-bare-dict"),
        pytest.param(Dict[str, List[int]], Dict[str, List], id="dict-with-list-to-dict-with-bare-list"),
        pytest.param(Set[Tuple[int, str]], Set[Tuple], id="set-of-tuple-to-set-of-bare-tuple"),
        pytest.param(
            Tuple[List[int], Dict[str, bool]], Tuple[List, Dict], id="tuple-of-containers-to-tuple-of-bare-containers"
        ),
        pytest.param(list[dict[str, int]], list[dict], id="list-of-dict-to-list-of-bare-dict"),
        pytest.param(dict[str, list[int]], dict[str, list], id="dict-with-list-to-dict-with-bare-list"),
        pytest.param(set[tuple[int, str]], set[tuple], id="set-of-tuple-to-set-of-bare-tuple"),
        pytest.param(
            tuple[list[int], dict[str, bool]], tuple[list, dict], id="tuple-of-containers-to-tuple-of-bare-containers"
        ),
        # Triple nesting
        pytest.param(List[List[List[int]]], List[List[List]], id="triple-nested-list-to-partially-bare"),
        pytest.param(List[List[List[int]]], List[List], id="triple-nested-list-to-double-bare"),
        pytest.param(List[List[List[int]]], List, id="triple-nested-list-to-completely-bare"),
        pytest.param(list[list[list[int]]], list[list[list]], id="triple-nested-list-to-partially-bare"),
        pytest.param(list[list[list[int]]], list[list], id="triple-nested-list-to-double-bare"),
        pytest.param(list[list[list[int]]], list, id="triple-nested-list-to-completely-bare"),
    ],
)
def test_nested_container_compatibility(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)
    # Bare container types should not be compatible with their typed counterparts
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(Callable[[int, str], bool], Callable, id="callable-to-bare-callable"),
        pytest.param(Callable[[List], int], Callable[[List], Any], id="callable-list-int-to-any"),
        pytest.param(Callable[[list], int], Callable[[list], Any], id="callable-list-int-to-any"),
    ],
)
def test_callable_compatibility(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(
            Callable[[Callable[[int], str]], List[str]], Callable, id="callable-with-nested-types-to-bare-callable"
        ),
        pytest.param(
            Callable[[Callable[[int], str]], List[str]],
            Callable[[Callable[[Any], Any]], Any],
            id="double-nested-callable",
        ),
        pytest.param(
            Callable[[Callable[[Callable[[int], str]], bool]], str],
            Callable[[Callable[[Callable[[Any], Any]], Any]], Any],
            id="triple-nested-callable",
        ),
    ],
)
def test_nested_callable_compatibility(sender_type, receiver_type):
    assert _types_are_compatible(sender_type, receiver_type)
    # Bare callable container types should not be compatible with their typed counterparts
    assert not _types_are_compatible(receiver_type, sender_type)


@pytest.mark.parametrize(
    "sender_type,receiver_type",
    [
        pytest.param(
            Callable[[int, str], bool], Callable[[int, str], List], id="callable-to-callable-with-different-return-type"
        ),
        pytest.param(
            Callable[[int, int], bool], Callable[[int, str], bool], id="callable-to-callable-with-different-args-type"
        ),
        pytest.param(
            Callable[[int, str], bool], Callable[[int, str, int], bool], id="callable-to-callable-with-different-args"
        ),
        pytest.param(Callable[[int, str], bool], Callable[[int], bool], id="callable-to-callable-with-fewer-args"),
    ],
)
def test_always_incompatible_callable_types(sender_type, receiver_type):
    assert not _types_are_compatible(sender_type, receiver_type)
    assert not _types_are_compatible(receiver_type, sender_type)
