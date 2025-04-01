from typing import Any, List, Optional, Union

from haystack.core.component.types import GreedyVariadic, Variadic
from haystack.core.super_component.utils import _is_compatible
from haystack.core.super_component.super_component import InvalidMappingTypeError, InvalidMappingValueError


class TestTypeCompatibility:
    """
    Test suite for type compatibility checking functionality.
    """

    def test_basic_types(self):
        """Test compatibility of basic Python types."""
        assert _is_compatible(str, str)
        assert _is_compatible(int, int)
        assert not _is_compatible(str, int)
        assert not _is_compatible(float, int)

    def test_any_type(self):
        """Test Any type compatibility."""
        assert _is_compatible(int, Any)
        assert _is_compatible(Any, int)
        assert _is_compatible(Any, Any)
        assert _is_compatible(Any, str)
        assert _is_compatible(str, Any)

    def test_union_types(self):
        """Test Union type compatibility."""
        assert _is_compatible(int, Union[int, str])
        assert _is_compatible(Union[int, str], int)
        assert _is_compatible(Union[int, str], Union[str, int])
        assert _is_compatible(str, Union[int, str])
        assert not _is_compatible(bool, Union[int, str])
        assert not _is_compatible(float, Union[int, str])

    def test_variadic_type_compatibility(self):
        """Test compatibility with Variadic and GreedyVariadic types."""
        # Basic type compatibility
        variadic_int = Variadic[int]
        greedy_int = GreedyVariadic[int]

        assert _is_compatible(variadic_int, int)
        assert _is_compatible(int, variadic_int)
        assert _is_compatible(greedy_int, int)
        assert _is_compatible(int, greedy_int)

        # List type compatibility
        variadic_list = Variadic[List[int]]
        greedy_list = GreedyVariadic[List[int]]

        assert _is_compatible(variadic_list, List[int])
        assert _is_compatible(List[int], variadic_list)
        assert _is_compatible(greedy_list, List[int])
        assert _is_compatible(List[int], greedy_list)

    def test_nested_type_unwrapping(self):
        """Test nested type unwrapping behavior with unwrap_nested parameter."""
        # Test with unwrap_nested=True (default)
        nested_optional = Variadic[List[Optional[int]]]
        assert _is_compatible(nested_optional, List[int])
        assert _is_compatible(List[int], nested_optional)

        nested_union = Variadic[List[Union[int, None]]]
        assert _is_compatible(nested_union, List[int])
        assert _is_compatible(List[int], nested_union)

    def test_complex_nested_types(self):
        """Test complex nested type scenarios."""
        # Multiple levels of nesting
        complex_type = Variadic[List[List[Variadic[int]]]]
        target_type = List[List[int]]

        # With unwrap_nested=True
        assert _is_compatible(complex_type, target_type)
        assert _is_compatible(target_type, complex_type)

        # With unwrap_nested=False
        assert not _is_compatible(complex_type, target_type, unwrap_nested=False)
        assert not _is_compatible(target_type, complex_type, unwrap_nested=False)

    def test_mixed_variadic_types(self):
        """Test mixing Variadic and GreedyVariadic with other type constructs."""
        # Variadic with Union
        var_union = Variadic[Union[int, str]]
        assert _is_compatible(var_union, Union[int, str])
        assert _is_compatible(Union[int, str], var_union)

        # GreedyVariadic with Optional
        greedy_opt = GreedyVariadic[Optional[int]]
        assert _is_compatible(greedy_opt, int)
        assert _is_compatible(int, greedy_opt)

        # Nested Variadic and GreedyVariadic
        nested_var = Variadic[List[GreedyVariadic[int]]]
        assert _is_compatible(nested_var, List[int])

    def test_error_type_compatibility(self):
        """Test compatibility of error types."""
        # Error types should be compatible with themselves
        assert _is_compatible(InvalidMappingTypeError, InvalidMappingTypeError)
        assert _is_compatible(InvalidMappingValueError, InvalidMappingValueError)

        # Error types should not be compatible with each other
        assert not _is_compatible(InvalidMappingTypeError, InvalidMappingValueError)
        assert not _is_compatible(InvalidMappingValueError, InvalidMappingTypeError)

        # Error types should be compatible with Exception
        assert _is_compatible(InvalidMappingTypeError, Exception)
        assert _is_compatible(InvalidMappingValueError, Exception)

        # Error types should be compatible with Any
        assert _is_compatible(InvalidMappingTypeError, Any)
        assert _is_compatible(InvalidMappingValueError, Any)

    def test_error_type_with_union(self):
        """Test error types in Union types."""
        error_union = Union[InvalidMappingTypeError, InvalidMappingValueError]

        # Should be compatible with both error types
        assert _is_compatible(error_union, InvalidMappingTypeError)
        assert _is_compatible(error_union, InvalidMappingValueError)

        # Should be compatible with Exception
        assert _is_compatible(error_union, Exception)

        # Should not be compatible with other types
        assert not _is_compatible(error_union, str)
        assert not _is_compatible(error_union, int)
