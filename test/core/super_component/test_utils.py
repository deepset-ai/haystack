from typing import Any, List, Optional, Tuple, Union, cast

from haystack.core.component.types import GreedyVariadic, Variadic
from haystack.core.super_component.utils import _is_compatible, get_args


class TestTypeCompatibility:
    """
    Test suite for type compatibility checking functionality.
    """

    def test_basic_types(self):
        """Test compatibility of basic Python types."""
        is_compat, common = _is_compatible(str, str)
        assert is_compat and common == str

        is_compat, common = _is_compatible(int, int)
        assert is_compat and common == int

        is_compat, common = _is_compatible(str, int)
        assert not is_compat and common is None

        is_compat, common = _is_compatible(float, int)
        assert not is_compat and common is None

    def test_any_type(self):
        """Test Any type compatibility."""
        is_compat, common = _is_compatible(int, Any)
        assert is_compat and common == int

        is_compat, common = _is_compatible(Any, int)
        assert is_compat and common == int

        is_compat, common = _is_compatible(Any, Any)
        assert is_compat and common == Any

        is_compat, common = _is_compatible(Any, str)
        assert is_compat and common == str

        is_compat, common = _is_compatible(str, Any)
        assert is_compat and common == str

    def test_union_types(self):
        """Test Union type compatibility."""
        is_compat, common = _is_compatible(int, Union[int, str])
        assert is_compat and common == int

        is_compat, common = _is_compatible(Union[int, str], int)
        assert is_compat and common == int

        is_compat, common = _is_compatible(Union[int, str], Union[str, int])
        assert is_compat and common == Union[int, str] or common == Union[str, int]

        is_compat, common = _is_compatible(str, Union[int, str])
        assert is_compat and common == str

        is_compat, common = _is_compatible(bool, Union[int, str])
        assert not is_compat and common is None

        is_compat, common = _is_compatible(float, Union[int, str])
        assert not is_compat and common is None

    def test_variadic_type_compatibility(self):
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
        variadic_list = Variadic[List[int]]
        greedy_list = GreedyVariadic[List[int]]

        is_compat, common = _is_compatible(variadic_list, List[int])
        assert is_compat and common == List[int]

        is_compat, common = _is_compatible(List[int], variadic_list)
        assert is_compat and common == List[int]

        is_compat, common = _is_compatible(greedy_list, List[int])
        assert is_compat and common == List[int]

        is_compat, common = _is_compatible(List[int], greedy_list)
        assert is_compat and common == List[int]

    def test_nested_type_unwrapping(self):
        """Test nested type unwrapping behavior with unwrap_nested parameter."""
        # Test with unwrap_nested=True (default)
        nested_optional = Variadic[List[Optional[int]]]

        is_compat, common = _is_compatible(nested_optional, List[int])
        assert is_compat and common == List[int]

        is_compat, common = _is_compatible(List[int], nested_optional)
        assert is_compat and common == List[int]

        nested_union = Variadic[List[Union[int, None]]]

        is_compat, common = _is_compatible(nested_union, List[int])
        assert is_compat and common == List[int]

        is_compat, common = _is_compatible(List[int], nested_union)
        assert is_compat and common == List[int]

    def test_complex_nested_types(self):
        """Test complex nested type scenarios."""
        # Multiple levels of nesting
        complex_type = Variadic[List[List[Variadic[int]]]]
        target_type = List[List[int]]

        # With unwrap_nested=True
        is_compat, common = _is_compatible(complex_type, target_type)
        assert is_compat and common == List[List[int]]

        is_compat, common = _is_compatible(target_type, complex_type)
        assert is_compat and common == List[List[int]]

        # With unwrap_nested=False
        is_compat, common = _is_compatible(complex_type, target_type, unwrap_nested=False)
        assert not is_compat and common is None

        is_compat, common = _is_compatible(target_type, complex_type, unwrap_nested=False)
        assert not is_compat and common is None

    def test_mixed_variadic_types(self):
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
        nested_var = Variadic[List[GreedyVariadic[int]]]

        is_compat, common = _is_compatible(nested_var, List[int])
        assert is_compat and common == List[int]
