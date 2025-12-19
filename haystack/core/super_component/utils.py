# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import UnionType
from typing import Annotated, Any, TypeVar, Union, cast, get_args, get_origin

from haystack.core.component.types import HAYSTACK_GREEDY_VARIADIC_ANNOTATION, HAYSTACK_VARIADIC_ANNOTATION
from haystack.utils.type_serialization import _build_pep604_union_type, _is_union_type


class _delegate_default:
    """Custom object for delegating filling of default values to the underlying components."""


T = TypeVar("T")


def _is_compatible(type1: T, type2: T, unwrap_nested: bool = True) -> tuple[bool, T | None]:
    """
    Check if two types are compatible (bidirectional/symmetric check).

    :param type1: First type to compare
    :param type2: Second type to compare
    :param unwrap_nested: If True, recursively unwraps nested Optional and Variadic types.
        If False, only unwraps at the top level.
    :return: Tuple of (True if types are compatible, common type if compatible)
    """
    type1_unwrapped = _unwrap_all(type1, recursive=unwrap_nested)
    type2_unwrapped = _unwrap_all(type2, recursive=unwrap_nested)

    return _types_are_compatible(type1_unwrapped, type2_unwrapped)


def _types_are_compatible(type1: T, type2: T) -> tuple[bool, T | None]:
    """
    Core type compatibility check implementing symmetric matching.

    :param type1: First unwrapped type to compare
    :param type2: Second unwrapped type to compare
    :return: True if types are compatible, False otherwise
    """
    # Handle Any type
    if type1 is Any:
        return True, type2
    if type2 is Any:
        return True, type1

    # Direct equality
    if type1 == type2:
        return True, type1

    type1_origin = get_origin(type1)
    type2_origin = get_origin(type2)

    # Handle Union types (including X | Y syntax)
    if _is_union_type(type1_origin) or _is_union_type(type2_origin):
        return _check_union_compatibility(type1, type2, type1_origin, type2_origin)

    # Handle non-Union types
    return _check_non_union_compatibility(type1, type2, type1_origin, type2_origin)


def _check_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> tuple[bool, T | None]:
    """Handle all Union type compatibility cases (including X | Y syntax)."""
    if _is_union_type(type1_origin) and not _is_union_type(type2_origin):
        # Find all compatible types from the union
        compatible_types = []
        for union_arg in get_args(type1):
            is_compat, common = _types_are_compatible(union_arg, type2)
            if is_compat and common is not None:
                compatible_types.append(common)
        if compatible_types:
            # The constructed Union or single type must be cast to T | None
            # to satisfy mypy, as T is specific to this function's call context.
            result_type = _build_pep604_union_type(compatible_types)
            return True, cast(T | None, result_type)
        return False, None

    if _is_union_type(type2_origin) and not _is_union_type(type1_origin):
        # Find all compatible types from the union
        compatible_types = []
        for union_arg in get_args(type2):
            is_compat, common = _types_are_compatible(type1, union_arg)
            if is_compat and common is not None:
                compatible_types.append(common)
        if compatible_types:
            # The constructed Union or single type must be cast to T | None
            # to satisfy mypy, as T is specific to this function's call context.
            result_type = _build_pep604_union_type(compatible_types)
            return True, cast(T | None, result_type)
        return False, None

    # Both are Union types
    compatible_types = []
    for arg1 in get_args(type1):
        for arg2 in get_args(type2):
            is_compat, common = _types_are_compatible(arg1, arg2)
            if is_compat and common is not None:
                compatible_types.append(common)

    if compatible_types:
        # The constructed Union or single type must be cast to T | None
        # to satisfy mypy, as T is specific to this function's call context.
        result_type = _build_pep604_union_type(compatible_types)
        return True, cast(T | None, result_type)
    return False, None


def _check_non_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> tuple[bool, T | None]:
    """Handle non-Union type compatibility cases."""
    # If no origin, compare types directly
    if not type1_origin and not type2_origin:
        if type1 == type2:
            return True, type1
        return False, None

    # Both must have origins and they must be equal
    if not (type1_origin and type2_origin and type1_origin == type2_origin):
        return False, None

    # Compare generic type arguments
    type1_args = get_args(type1)
    type2_args = get_args(type2)

    if len(type1_args) != len(type2_args):
        return False, None

    # Check if all arguments are compatible
    common_args = []
    for t1_arg, t2_arg in zip(type1_args, type2_args):
        is_compat, common = _types_are_compatible(t1_arg, t2_arg)
        if not is_compat:
            return False, None
        common_args.append(common)

    # Reconstruct the type with common arguments
    return True, cast(T | None, type1_origin[tuple(common_args)])


def _unwrap_all(t: T, recursive: bool) -> T:
    """
    Unwrap a type until no more unwrapping is possible.

    :param t: Type to unwrap
    :param recursive: If True, recursively unwraps nested types
    :return: The fully unwrapped type
    """
    # First handle top-level Variadic/GreedyVariadic
    if _is_variadic_type(t):
        t = _unwrap_variadics(t, recursive=recursive)
    else:
        # If it's a generic type and we're unwrapping recursively
        origin = get_origin(t)
        if recursive and origin is not None and (args := get_args(t)):
            unwrapped_args = tuple(_unwrap_all(arg, recursive) for arg in args)
            # types.UnionType (PEP 604 X | Y) is not subscriptable, so we use typing.Union instead
            if origin is UnionType:
                t = cast(T, Union[unwrapped_args])
            else:
                t = origin[unwrapped_args]

    # Then handle top-level Optional
    if _is_optional_type(t):
        t = _unwrap_optionals(t, recursive=recursive)

    return t


def _is_variadic_type(t: T) -> bool:
    """Check if type is a Variadic or GreedyVariadic type."""
    origin = get_origin(t)
    if origin is Annotated:
        args = get_args(t)
        return len(args) >= 2 and args[1] in (HAYSTACK_VARIADIC_ANNOTATION, HAYSTACK_GREEDY_VARIADIC_ANNOTATION)  # noqa: PLR2004
    return False


def _is_optional_type(t: T) -> bool:
    """Check if type is an Optional type (Union[X, None] or X | None)."""
    origin = get_origin(t)
    if _is_union_type(origin):
        args = get_args(t)
        return type(None) in args
    return False


def _unwrap_variadics(t: T, recursive: bool) -> T:
    """
    Unwrap Variadic or GreedyVariadic annotated types.

    :param t: Type to unwrap
    :param recursive: If True, recursively unwraps nested types
    :return: Unwrapped type if it was a variadic type, original type otherwise
    """
    if not _is_variadic_type(t):
        return t

    args = get_args(t)
    # Get the Iterable[X] type and extract X
    iterable_type = args[0]
    inner_type = get_args(iterable_type)[0]

    # Only recursively unwrap if requested
    if recursive:
        return _unwrap_all(inner_type, recursive)
    return inner_type


def _unwrap_optionals(t: T, recursive: bool) -> T:
    """
    Unwrap Optional[...] types (Union[X, None]).

    :param t: Type to unwrap
    :param recursive: If True, recursively unwraps nested types
    :return: Unwrapped type if it was an Optional, original type otherwise
    """
    if not _is_optional_type(t):
        return t

    args = list(get_args(t))
    args.remove(type(None))
    result = args[0] if len(args) == 1 else Union[tuple(args)]

    # Only recursively unwrap if requested
    if recursive:
        return _unwrap_all(result, recursive)  # type: ignore
    return result  # type: ignore
