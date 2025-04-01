# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

from haystack.core.component.types import HAYSTACK_GREEDY_VARIADIC_ANNOTATION, HAYSTACK_VARIADIC_ANNOTATION


class _delegate_default:
    """Custom object for delegating filling of default values to the underlying components."""


T = TypeVar("T")


def _is_compatible(type1: T, type2: T, unwrap_nested: bool = True) -> bool:
    """
    Check if two types are compatible (bidirectional/symmetric check).

    :param type1: First type to compare
    :param type2: Second type to compare
    :param unwrap_nested: If True, recursively unwraps nested Optional and Variadic types.
        If False, only unwraps at the top level.
    :return: True if types are compatible, False otherwise
    """
    type1_unwrapped = _unwrap_all(type1, recursive=unwrap_nested)
    type2_unwrapped = _unwrap_all(type2, recursive=unwrap_nested)

    return _types_are_compatible(type1_unwrapped, type2_unwrapped)


def _types_are_compatible(type1: T, type2: T) -> bool:
    """
    Core type compatibility check implementing symmetric matching.

    :param type1: First unwrapped type to compare
    :param type2: Second unwrapped type to compare
    :return: True if types are compatible, False otherwise
    """
    # Handle Any type and direct equality
    if type1 is Any or type2 is Any or type1 == type2:
        return True

    type1_origin = get_origin(type1)
    type2_origin = get_origin(type2)

    # Handle Union types
    if type1_origin is Union or type2_origin is Union:
        return _check_union_compatibility(type1, type2, type1_origin, type2_origin)

    # Handle non-Union types
    return _check_non_union_compatibility(type1, type2, type1_origin, type2_origin)


def _check_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> bool:
    """Handle all Union type compatibility cases."""
    if type1_origin is Union and type2_origin is not Union:
        return any(_types_are_compatible(union_arg, type2) for union_arg in get_args(type1))
    if type2_origin is Union and type1_origin is not Union:
        return any(_types_are_compatible(type1, union_arg) for union_arg in get_args(type2))
    # Both are Union types. Check all type combinations are compatible.
    return any(any(_types_are_compatible(arg1, arg2) for arg2 in get_args(type2)) for arg1 in get_args(type1))


def _check_non_union_compatibility(type1: T, type2: T, type1_origin: Any, type2_origin: Any) -> bool:
    """Handle non-Union type compatibility cases."""
    # If no origin, compare types directly
    if not type1_origin and not type2_origin:
        return type1 == type2

    # Both must have origins and they must be equal
    if not (type1_origin and type2_origin and type1_origin == type2_origin):
        return False

    # Compare generic type arguments
    type1_args = get_args(type1)
    type2_args = get_args(type2)

    if len(type1_args) != len(type2_args):
        return False

    return all(_types_are_compatible(t1_arg, t2_arg) for t1_arg, t2_arg in zip(type1_args, type2_args))


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
    """Check if type is an Optional type."""
    origin = get_origin(t)
    if origin is Union:
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
    result = args[0] if len(args) == 1 else Union[tuple(args)]  # type: ignore

    # Only recursively unwrap if requested
    if recursive:
        return _unwrap_all(result, recursive)  # type: ignore
    return result  # type: ignore
