import pytest
from typing import List, Dict, Optional, Union, TypeVar, Generic
from dataclasses import dataclass

from haystack.dataclasses.state_utils import _is_list_type, merge_lists, _is_valid_type

import inspect


def test_is_list_type():
    assert _is_list_type(list) is True
    assert _is_list_type(List[int]) is True
    assert _is_list_type(List[str]) is True
    assert _is_list_type(dict) is False
    assert _is_list_type(int) is False
    assert _is_list_type(Union[List[int], None]) is False


class TestMergeLists:
    def test_merge_two_lists(self):
        current = [1, 2, 3]
        new = [4, 5, 6]
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4, 5, 6]
        # Ensure original lists weren't modified
        assert current == [1, 2, 3]
        assert new == [4, 5, 6]

    def test_append_to_list(self):
        current = [1, 2, 3]
        new = 4
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4]
        assert current == [1, 2, 3]  # Ensure original wasn't modified

    def test_create_new_list(self):
        current = 1
        new = 2
        result = merge_lists(current, new)
        assert result == [1, 2]

    def test_replace_with_list(self):
        current = 1
        new = [2, 3]
        result = merge_lists(current, new)
        assert result == [1, 2, 3]


class TestIsValidType:
    def test_builtin_types(self):
        assert _is_valid_type(str) is True
        assert _is_valid_type(int) is True
        assert _is_valid_type(dict) is True
        assert _is_valid_type(list) is True
        assert _is_valid_type(tuple) is True
        assert _is_valid_type(set) is True
        assert _is_valid_type(bool) is True
        assert _is_valid_type(float) is True

    def test_generic_types(self):
        assert _is_valid_type(List[str]) is True
        assert _is_valid_type(Dict[str, int]) is True
        assert _is_valid_type(List[Dict[str, int]]) is True
        assert _is_valid_type(Dict[str, List[int]]) is True

    def test_custom_classes(self):
        @dataclass
        class CustomClass:
            value: int

        T = TypeVar("T")

        class GenericCustomClass(Generic[T]):
            pass

        # Test regular and generic custom classes
        assert _is_valid_type(CustomClass) is True
        assert _is_valid_type(GenericCustomClass) is True
        assert _is_valid_type(GenericCustomClass[int]) is True

        # Test generic types with custom classes
        assert _is_valid_type(List[CustomClass]) is True
        assert _is_valid_type(Dict[str, CustomClass]) is True
        assert _is_valid_type(Dict[str, GenericCustomClass[int]]) is True

    def test_invalid_types(self):
        # Test regular values
        assert _is_valid_type(42) is False
        assert _is_valid_type("string") is False
        assert _is_valid_type([1, 2, 3]) is False
        assert _is_valid_type({"a": 1}) is False
        assert _is_valid_type(True) is False

        # Test class instances
        @dataclass
        class SampleClass:
            value: int

        instance = SampleClass(42)
        assert _is_valid_type(instance) is False

        # Test callable objects
        assert _is_valid_type(len) is False
        assert _is_valid_type(lambda x: x) is False
        assert _is_valid_type(print) is False

    def test_union_and_optional_types(self):
        # Test basic Union types
        assert _is_valid_type(Union[str, int]) is True
        assert _is_valid_type(Union[str, None]) is True
        assert _is_valid_type(Union[List[int], Dict[str, str]]) is True

        # Test Optional types (which are Union[T, None])
        assert _is_valid_type(Optional[str]) is True
        assert _is_valid_type(Optional[List[int]]) is True
        assert _is_valid_type(Optional[Dict[str, list]]) is True

        # Test that Union itself is not a valid type (only instantiated Unions are)
        assert _is_valid_type(Union) is False

    def test_nested_generic_types(self):
        assert _is_valid_type(List[List[Dict[str, List[int]]]]) is True
        assert _is_valid_type(Dict[str, List[Dict[str, set]]]) is True
        assert _is_valid_type(Dict[str, Optional[List[int]]]) is True
        assert _is_valid_type(List[Union[str, Dict[str, List[int]]]]) is True

    def test_edge_cases(self):
        # Test None and NoneType
        assert _is_valid_type(None) is False
        assert _is_valid_type(type(None)) is True

        # Test functions and methods
        def sample_func():
            pass

        assert _is_valid_type(sample_func) is False
        assert _is_valid_type(type(sample_func)) is True

        # Test modules
        assert _is_valid_type(inspect) is False

        # Test type itself
        assert _is_valid_type(type) is True

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (str, True),
            (int, True),
            (List[int], True),
            (Dict[str, int], True),
            (Union[str, int], True),
            (Optional[str], True),
            (42, False),
            ("string", False),
            ([1, 2, 3], False),
            (lambda x: x, False),
        ],
    )
    def test_parametrized_cases(self, test_input, expected):
        assert _is_valid_type(test_input) is expected
