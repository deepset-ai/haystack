# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import pytest

from haystack.core.component.types import GreedyVariadic, Variadic
from haystack.core.super_component.utils import _is_compatible


@pytest.mark.parametrize(
    "left,right,expected_common",
    [
        (str, str, str),
        (int, int, int),
        (float, float, float),
        (bool, bool, bool),
        (list, list, list),
        (dict, dict, dict),
        (set, set, set),
        (tuple, tuple, tuple),
    ],
)
def test_basic_types_compatible(left, right, expected_common):
    """Test compatible basic Python types."""
    is_compat, common = _is_compatible(left, right)
    assert is_compat
    assert common == expected_common


@pytest.mark.parametrize("left,right", [(str, int), (float, int), (bool, str), (list, dict), (set, tuple)])
def test_basic_types_incompatible(left, right):
    """Test incompatible basic Python types."""
    is_compat, common = _is_compatible(left, right)
    assert not is_compat
    assert common is None


@pytest.mark.parametrize(
    "left,right,expected_common",
    [
        (Any, Any, Any),
        # int
        (int, Any, int),
        (Any, int, int),
        # str
        (Any, str, str),
        (str, Any, str),
        # float
        (float, Any, float),
        (Any, float, float),
        # list
        (list, Any, list),
        (Any, list, list),
        # dict
        (dict, Any, dict),
        (Any, dict, dict),
    ],
)
def test_any_type(left, right, expected_common):
    """Test Any type compatibility."""
    is_compat, common = _is_compatible(left, right)
    assert is_compat
    assert common == expected_common


def test_union_types():
    """Test Union type compatibility."""
    is_compat, common = _is_compatible(int, Union[int, str])
    assert is_compat and common == int

    is_compat, common = _is_compatible(Union[int, str], int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(Union[int, str], Union[str, int])
    assert is_compat and common == Union[int, str]

    is_compat, common = _is_compatible(str, Union[int, str])
    assert is_compat and common == str

    is_compat, common = _is_compatible(bool, Union[int, str])
    assert not is_compat and common is None

    is_compat, common = _is_compatible(float, Union[int, str])
    assert not is_compat and common is None

    # PEP 604 union types (X | Y syntax)
    is_compat, common = _is_compatible(int, int | str)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int | str, int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int | str, str | int)
    assert is_compat and (common == int | str or common == str | int)

    is_compat, common = _is_compatible(str, str | None)
    assert is_compat and common == str

    is_compat, common = _is_compatible(str | None, str)
    assert is_compat and common == str

    is_compat, common = _is_compatible(bool, int | str)
    assert not is_compat and common is None

    is_compat, common = _is_compatible(float, int | str)
    assert not is_compat and common is None

    # PEP 604 with typing.Union
    is_compat, common = _is_compatible(int | str, Union[int, str])
    assert is_compat and common == int | str

    is_compat, common = _is_compatible(Union[int, str], int | str)
    assert is_compat and common == Union[int, str]

    is_compat, common = _is_compatible(int | str, int | str)
    assert is_compat and common == int | str

    is_compat, common = _is_compatible(str | None, Optional[str])
    assert is_compat and common == str | None

    is_compat, common = _is_compatible(Optional[str], str | None)
    assert is_compat and common == Optional[str]

    is_compat, common = _is_compatible(str | None, str | None)
    assert is_compat and common == str | None


def test_variadic_type_compatibility():
    """Test compatibility with Variadic and GreedyVariadic types."""
    # Basic type compatibility
    variadic_int = Variadic[int]
    greedy_int = GreedyVariadic[int]

    is_compat, common = _is_compatible(variadic_int, int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int, variadic_int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(greedy_int, int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int, greedy_int)
    assert is_compat and common == int

    # List type compatibility
    variadic_list = Variadic[list[int]]
    greedy_list = GreedyVariadic[list[int]]

    is_compat, common = _is_compatible(variadic_list, list[int])
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(list[int], variadic_list)
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(greedy_list, list[int])
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(list[int], greedy_list)
    assert is_compat and common == list[int]

    # PEP 604 with Variadic
    variadic_union = Variadic[int | str]

    is_compat, common = _is_compatible(variadic_union, int | str)
    assert is_compat and common == int | str

    is_compat, common = _is_compatible(int | str, variadic_union)
    assert is_compat and common == int | str

    # PEP 604 optional with GreedyVariadic
    greedy_opt = GreedyVariadic[int | None]

    is_compat, common = _is_compatible(greedy_opt, int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int, greedy_opt)
    assert is_compat and common == int


def test_nested_type_unwrapping():
    """Test nested type unwrapping behavior with unwrap_nested parameter."""
    # Test with unwrap_nested=True (default)
    nested_optional = Variadic[list[int | None]]

    is_compat, common = _is_compatible(nested_optional, list[int])
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(list[int], nested_optional)
    assert is_compat and common == list[int]

    nested_union = Variadic[list[Union[int, None]]]

    is_compat, common = _is_compatible(nested_union, list[int])
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(list[int], nested_union)
    assert is_compat and common == list[int]

    # PEP 604 (X | Y and X | None syntax)
    nested_pep604_optional = Variadic[list[int | None]]

    is_compat, common = _is_compatible(nested_pep604_optional, list[int])
    assert is_compat and common == list[int]

    is_compat, common = _is_compatible(list[int], nested_pep604_optional)
    assert is_compat and common == list[int]

    nested_pep604_union = Variadic[list[str | int]]

    is_compat, common = _is_compatible(nested_pep604_union, list[str | int])
    assert is_compat and common == list[str | int]


def test_complex_nested_types():
    """Test complex nested type scenarios."""
    # Multiple levels of nesting
    complex_type = Variadic[list[list[Variadic[int]]]]
    target_type = list[list[int]]

    # With unwrap_nested=True
    is_compat, common = _is_compatible(complex_type, target_type)
    assert is_compat and common == list[list[int]]

    is_compat, common = _is_compatible(target_type, complex_type)
    assert is_compat and common == list[list[int]]

    # With unwrap_nested=False
    is_compat, common = _is_compatible(complex_type, target_type, unwrap_nested=False)
    assert not is_compat and common is None

    is_compat, common = _is_compatible(target_type, complex_type, unwrap_nested=False)
    assert not is_compat and common is None


def test_mixed_variadic_types():
    """Test mixing Variadic and GreedyVariadic with other type constructs."""
    # Variadic with Union
    var_union = Variadic[Union[int, str]]

    is_compat, common = _is_compatible(var_union, Union[int, str])
    assert is_compat and common == Union[int, str]

    is_compat, common = _is_compatible(Union[int, str], var_union)
    assert is_compat and common == Union[int, str]

    # GreedyVariadic with Optional
    greedy_opt = GreedyVariadic[Optional[int]]

    is_compat, common = _is_compatible(greedy_opt, int)
    assert is_compat and common == int

    is_compat, common = _is_compatible(int, greedy_opt)
    assert is_compat and common == int

    # Nested Variadic and GreedyVariadic
    nested_var = Variadic[list[GreedyVariadic[int]]]

    is_compat, common = _is_compatible(nested_var, list[int])
    assert is_compat and common == list[int]
