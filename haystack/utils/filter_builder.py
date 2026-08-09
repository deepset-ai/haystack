# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from haystack.errors import FilterError

# Operator strings are spelled out rather than unpacked from COMPARISON_OPERATORS, whose
# iteration order is an implementation detail: reordering that dict must not silently
# remap `gt` onto "<". test_filter_builder.py asserts this list stays in sync with
# COMPARISON_OPERATORS, the single source of truth used by `document_matches_filter`.
_EQ, _NE, _GT, _GTE, _LT, _LTE, _IN, _NOT_IN = ("==", "!=", ">", ">=", "<", "<=", "in", "not in")


class FilterBuilder:
    """
    Helper class to build Haystack metadata filter dictionaries with a fluent interface.

    The builder emits the same dictionary format that `document_matches_filter` and the
    DocumentStore protocol already understand: a comparison condition is
    `{"field": ..., "operator": ..., "value": ...}` and a logical condition is
    `{"operator": "AND"|"OR"|"NOT", "conditions": [...]}`.

    Example:
    ```python
    from haystack.utils import FilterBuilder

    filters = (
        FilterBuilder()
        .eq("meta.type", "article")
        .or_group(lambda f: f.in_("meta.genre", ["economy"]).eq("meta.publisher", "nytimes"))
        .build()
    )
    ```
    """

    def __init__(self) -> None:
        self._conditions: list[dict[str, Any]] = []

    def _add_condition(self, field: str, operator: str, value: Any) -> FilterBuilder:
        self._conditions.append({"field": field, "operator": operator, "value": value})
        return self

    def eq(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be equal to `value`.
        """
        return self._add_condition(field, _EQ, value)

    def ne(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must not be equal to `value`.
        """
        return self._add_condition(field, _NE, value)

    def gt(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be greater than `value`.
        """
        return self._add_condition(field, _GT, value)

    def gte(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be greater than or equal to `value`.
        """
        return self._add_condition(field, _GTE, value)

    def lt(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be less than `value`.
        """
        return self._add_condition(field, _LT, value)

    def lte(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be less than or equal to `value`.
        """
        return self._add_condition(field, _LTE, value)

    def in_(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must be one of the values contained in `value`.
        """
        return self._add_condition(field, _IN, value)

    def not_in(self, field: str, value: Any) -> FilterBuilder:
        """
        Add a condition that `field` must not be any of the values contained in `value`.
        """
        return self._add_condition(field, _NOT_IN, value)

    def _add_group(self, operator: str, builder: Callable[[FilterBuilder], Any]) -> FilterBuilder:
        """
        Add a logical group condition built by `builder`.
        """
        sub_builder = FilterBuilder()
        builder(sub_builder)
        self._conditions.append({"operator": operator, "conditions": sub_builder._conditions})
        return self

    def and_group(self, builder: Callable[[FilterBuilder], Any]) -> FilterBuilder:
        """
        Add a logical AND group. `builder` receives a fresh FilterBuilder to fill with conditions.
        """
        return self._add_group("AND", builder)

    def or_group(self, builder: Callable[[FilterBuilder], Any]) -> FilterBuilder:
        """
        Add a logical OR group. `builder` receives a fresh FilterBuilder to fill with conditions.
        """
        return self._add_group("OR", builder)

    def not_group(self, builder: Callable[[FilterBuilder], Any]) -> FilterBuilder:
        """
        Add a logical NOT group. `builder` receives a fresh FilterBuilder to fill with conditions.
        """
        return self._add_group("NOT", builder)

    def build(self) -> dict[str, Any]:
        """
        Build the filter dictionary.

        A single condition is returned unwrapped. Multiple conditions are combined with a
        top-level logical "AND". Raises `FilterError` if no condition has been added.
        """
        if not self._conditions:
            msg = "The FilterBuilder contains no conditions. Add one before calling build()."
            raise FilterError(msg)
        if len(self._conditions) == 1:
            return self._conditions[0]
        return {"operator": "AND", "conditions": self._conditions}
