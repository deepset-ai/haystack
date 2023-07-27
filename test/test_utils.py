from typing import List, Set, Sequence, Tuple, Dict, Mapping, Literal, Union, Optional, Any
from enum import Enum
from pathlib import Path

import pytest

from canals.utils import _type_name


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
    "type_,repr",
    [
        pytest.param(str, "str", id="primitive-types"),
        pytest.param(Any, "Any", id="any"),
        pytest.param(TestClass1, "TestClass1", id="class"),
        pytest.param(Optional[int], "Optional[int]", id="shallow-optional-with-primitive"),
        pytest.param(Optional[Any], "Optional[Any]", id="shallow-optional-with-any"),
        pytest.param(Optional[TestClass1], "Optional[TestClass1]", id="shallow-optional-with-class"),
        pytest.param(Union[bool, TestClass1], "Union[bool, TestClass1]", id="shallow-union"),
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
        pytest.param(List[TestClass1], "List[TestClass1]", id="shallow-sequence-of-classes"),
        pytest.param(
            List[Set[Sequence[TestClass1]]], "List[Set[Sequence[TestClass1]]]", id="nested-sequence-of-classes"
        ),
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
        pytest.param(Dict[str, TestClass1], "Dict[str, TestClass1]", id="shallow-mapping-of-classes"),
        pytest.param(
            Dict[str, Mapping[str, Dict[str, TestClass1]]],
            "Dict[str, Mapping[str, Dict[str, TestClass1]]]",
            id="nested-mapping-of-classes",
        ),
        pytest.param(
            Literal["a", "b", "c"],
            "Literal['a', 'b', 'c']",
            id="string-literal",
        ),
        pytest.param(
            Literal[1, 2, 3],
            "Literal[1, 2, 3]",
            id="primitive-literal",
        ),
        pytest.param(
            Literal[TestEnum.TEST1],
            "Literal[TestEnum.TEST1]",
            id="enum-literal",
        ),
        pytest.param(
            Tuple[Optional[Literal["a", "b", "c"]], Union[Path, Dict[int, TestClass1]]],
            "Tuple[Optional[Literal['a', 'b', 'c']], Union[Path, Dict[int, TestClass1]]]",
            id="deeply-nested-complex-type",
        ),
    ],
)
def test_type_name(type_, repr):
    assert _type_name(type_) == repr
