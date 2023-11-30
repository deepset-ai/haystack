from typing import List, Set, Sequence, Tuple, Dict, Mapping, Literal, Union, Optional, Any
from enum import Enum
from pathlib import Path

import pytest

from haystack.core.type_utils import _type_name


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
    "type_,repr",
    [
        pytest.param(str, "str", id="primitive-types"),
        pytest.param(Any, "Any", id="any"),
        pytest.param(Class1, "Class1", id="class"),
        pytest.param(Optional[int], "Optional[int]", id="shallow-optional-with-primitive"),
        pytest.param(Optional[Any], "Optional[Any]", id="shallow-optional-with-any"),
        pytest.param(Optional[Class1], "Optional[Class1]", id="shallow-optional-with-class"),
        pytest.param(Union[bool, Class1], "Union[bool, Class1]", id="shallow-union"),
        pytest.param(List[str], "List[str]", id="shallow-sequence-of-primitives"),
        pytest.param(List[Set[Sequence[str]]], "List[Set[Sequence[str]]]", id="nested-sequence-of-primitives"),
        pytest.param(
            Optional[List[Set[Sequence[str]]]],
            "Optional[List[Set[Sequence[str]]]]",
            id="optional-nested-sequence-of-primitives",
        ),
        pytest.param(
            List[Set[Sequence[Optional[str]]]],
            "List[Set[Sequence[Optional[str]]]]",
            id="nested-optional-sequence-of-primitives",
        ),
        pytest.param(List[Class1], "List[Class1]", id="shallow-sequence-of-classes"),
        pytest.param(List[Set[Sequence[Class1]]], "List[Set[Sequence[Class1]]]", id="nested-sequence-of-classes"),
        pytest.param(Dict[str, int], "Dict[str, int]", id="shallow-mapping-of-primitives"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, int]]],
            "Dict[str, Mapping[str, Dict[str, int]]]",
            id="nested-mapping-of-primitives",
        ),
        pytest.param(
            Dict[str, Mapping[Any, Dict[str, int]]],
            "Dict[str, Mapping[Any, Dict[str, int]]]",
            id="nested-mapping-of-primitives-with-any",
        ),
        pytest.param(Dict[str, Class1], "Dict[str, Class1]", id="shallow-mapping-of-classes"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, Class1]]],
            "Dict[str, Mapping[str, Dict[str, Class1]]]",
            id="nested-mapping-of-classes",
        ),
        pytest.param(Literal["a", "b", "c"], "Literal['a', 'b', 'c']", id="string-literal"),
        pytest.param(Literal[1, 2, 3], "Literal[1, 2, 3]", id="primitive-literal"),
        pytest.param(Literal[Enum1.TEST1], "Literal[Enum1.TEST1]", id="enum-literal"),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, Class1]]],
            "Tuple[Optional[Literal['a', 'b', 'c']], Union[Path, Dict[int, Class1]]]",
            id="deeply-nested-complex-type",
        ),
    ],
)
def test_type_name(type_, repr):
    assert _type_name(type_) == repr
