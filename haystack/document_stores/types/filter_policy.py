# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, Literal, Optional

from haystack import logging

logger = logging.getLogger(__name__)


class FilterPolicy(Enum):
    """
    Policy to determine how filters are applied in retrievers interacting with document stores.
    """

    # Runtime filters replace init filters during retriever run invocation.
    REPLACE = "replace"

    # Runtime filters are merged with init filters, with runtime filters overwriting init values.
    MERGE = "merge"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(filter_policy: str) -> "FilterPolicy":
        """
        Convert a string to a FilterPolicy enum.

        :param filter_policy: The string to convert.
        :return: The corresponding FilterPolicy enum.
        """
        enum_map = {e.value.lower(): e for e in FilterPolicy}
        policy = enum_map.get(filter_policy.lower() if filter_policy else "")
        if policy is None:
            msg = f"Unknown FilterPolicy type '{filter_policy}'. Supported types are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return policy


def is_comparison_filter(filter_item: Dict[str, Any]) -> bool:
    """
    Check if the given filter is a comparison filter.

    :param filter_item: The filter to check.
    :returns: True if the filter is a comparison filter, False otherwise.
    """
    return all(key in filter_item for key in ["field", "operator", "value"])


def is_logical_filter(filter_item: Dict[str, Any]) -> bool:
    """
    Check if the given filter is a logical filter.

    :param filter_item: The filter to check.
    :returns: True if the filter is a logical filter, False otherwise.
    """
    return "operator" in filter_item and "conditions" in filter_item


def combine_two_logical_filters(
    init_logical_filter: Dict[str, Any], runtime_logical_filter: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine two logical filters, they must have the same operator.

    If `init_logical_filter["operator"]` and `runtime_logical_filter["operator"]` are the same, the conditions
    of both filters are combined. Otherwise, the `init_logical_filter` is ignored and `
    runtime_logical_filter` is returned.

        __Example__:

        ```python
        init_logical_filter = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
            ]
        }
        runtime_logical_filter = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
                {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
            ]
        }
        new_filters = combine_two_logical_filters(
            init_logical_filter, runtime_logical_filter, "AND"
        )
        # Output:
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {"field": "meta.genre", "operator": "IN", "value": ["economy", "politics"]},
                {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
            ]
        }
        ```
    """
    if init_logical_filter["operator"] == runtime_logical_filter["operator"]:
        return {
            "operator": str(init_logical_filter["operator"]),
            "conditions": init_logical_filter["conditions"] + runtime_logical_filter["conditions"],
        }

    logger.warning(
        "The provided logical operators, {parsed_operator} and {operator}, do not match so the parsed logical "
        "filter, {init_logical_filter}, will be ignored and only the provided logical filter,{runtime_logical_filter}, "
        "will be used. Update the logical operators to match to include the parsed filter.",
        parsed_operator=init_logical_filter["operator"],
        operator=runtime_logical_filter["operator"],
        init_logical_filter=init_logical_filter,
        runtime_logical_filter=runtime_logical_filter,
    )
    runtime_logical_filter["operator"] = str(runtime_logical_filter["operator"])
    return runtime_logical_filter


def combine_init_comparison_and_runtime_logical_filters(
    init_comparison_filter: Dict[str, Any],
    runtime_logical_filter: Dict[str, Any],
    logical_operator: Literal["AND", "OR", "NOT"],
) -> Dict[str, Any]:
    """
    Combine a runtime logical filter with the init comparison filter using the provided logical_operator.

    We only add the init_comparison_filter if logical_operator matches the existing
    runtime_logical_filter["operator"]. Otherwise, we return the runtime_logical_filter unchanged.

    __Example__:

    ```python
    runtime_logical_filter = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ]
    }
    init_comparison_filter = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    new_filters = combine_init_comparison_and_runtime_logical_filters(
        init_comparison_filter, runtime_logical_filter, "AND"
    )
    # Output:
    {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
        ]
    }
    ```
    """
    if runtime_logical_filter["operator"] == logical_operator:
        conditions = runtime_logical_filter["conditions"]
        fields = {c.get("field") for c in conditions}
        if init_comparison_filter["field"] not in fields:
            conditions.append(init_comparison_filter)
        else:
            logger.warning(
                "The init filter, {init_filter}, is ignored as the field is already present in the existing "
                "filters, {filters}.",
                init_filter=init_comparison_filter,
                filters=runtime_logical_filter,
            )
        return {"operator": str(runtime_logical_filter["operator"]), "conditions": conditions}

    logger.warning(
        "The provided logical_operator, {logical_operator}, does not match the logical operator found in "
        "the runtime filters, {filters_logical_operator}, so the init filter will be ignored.",
        logical_operator=logical_operator,
        filters_logical_operator=runtime_logical_filter["operator"],
    )
    runtime_logical_filter["operator"] = str(runtime_logical_filter["operator"])
    return runtime_logical_filter


def combine_runtime_comparison_and_init_logical_filters(
    runtime_comparison_filter: Dict[str, Any],
    init_logical_filter: Dict[str, Any],
    logical_operator: Literal["AND", "OR", "NOT"],
) -> Dict[str, Any]:
    """
    Combine an init logical filter with the runtime comparison filter using the provided logical_operator.

    We only add the runtime_comparison_filter if logical_operator matches the existing
    init_logical_filter["operator"]. Otherwise, we return the runtime_comparison_filter unchanged.

    __Example__:

    ```python
    init_logical_filter = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
        ]
    }
    runtime_comparison_filter = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"}
    new_filters = combine_runtime_comparison_and_init_logical_filters(
        runtime_comparison_filter, init_logical_filter, "AND"
    )
    # Output:
    {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": "article"},
            {"field": "meta.rating", "operator": ">=", "value": 3},
            {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
        ]
    }
    ```
    """
    if init_logical_filter["operator"] == logical_operator:
        conditions = init_logical_filter["conditions"]
        fields = {c.get("field") for c in conditions}
        if runtime_comparison_filter["field"] in fields:
            logger.warning(
                "The runtime filter, {runtime_filter}, will overwrite the existing filter with the same "
                "field in the init logical filter.",
                runtime_filter=runtime_comparison_filter,
            )
            conditions = [c for c in conditions if c.get("field") != runtime_comparison_filter["field"]]
        conditions.append(runtime_comparison_filter)
        return {"operator": str(init_logical_filter["operator"]), "conditions": conditions}

    logger.warning(
        "The provided logical_operator, {logical_operator}, does not match the logical operator found in "
        "the init logical filter, {filters_logical_operator}, so the init logical filter will be ignored.",
        logical_operator=logical_operator,
        filters_logical_operator=init_logical_filter["operator"],
    )
    return runtime_comparison_filter


def combine_two_comparison_filters(
    init_comparison_filter: Dict[str, Any],
    runtime_comparison_filter: Dict[str, Any],
    logical_operator: Literal["AND", "OR", "NOT"],
) -> Dict[str, Any]:
    """
    Combine a comparison filter with the `init_comparison_filter` using the provided `logical_operator`.

    If `runtime_comparison_filter` and `init_comparison_filter` target the same field, `init_comparison_filter`
    is ignored and `runtime_comparison_filter` is returned unchanged.

        __Example__:

        ```python
        runtime_comparison_filter = {"field": "meta.type", "operator": "==", "value": "article"},
        init_comparison_filter = {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
        new_filters = combine_two_comparison_filters(
            init_comparison_filter, runtime_comparison_filter, "AND"
        )
        # Output:
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": "2015-01-01"},
            ]
        }
        ```
    """
    if runtime_comparison_filter["field"] == init_comparison_filter["field"]:
        logger.warning(
            "The parsed filter, {parsed_filter}, is ignored as the field is already present in the existing "
            "filters, {filters}.",
            parsed_filter=init_comparison_filter,
            filters=runtime_comparison_filter,
        )
        return runtime_comparison_filter

    return {"operator": str(logical_operator), "conditions": [init_comparison_filter, runtime_comparison_filter]}


def apply_filter_policy(
    filter_policy: FilterPolicy,
    init_filters: Optional[Dict[str, Any]] = None,
    runtime_filters: Optional[Dict[str, Any]] = None,
    default_logical_operator: Literal["AND", "OR", "NOT"] = "AND",
) -> Optional[Dict[str, Any]]:
    """
    Apply the filter policy to the given initial and runtime filters to determine the final set of filters used.

    The function combines or replaces the initial and runtime filters based on the specified filter policy.

    :param filter_policy: The policy to apply when handling the filters. It can be one of the following:
        - `FilterPolicy.REPLACE`: Runtime filters will replace the initial filters.
        - `FilterPolicy.MERGE`: Runtime filters will be merged with the initial filters. If there are overlapping keys,
          values from the runtime filters will overwrite those from the initial filters.
    :param init_filters: The initial filters set during the initialization of the relevant retriever.
    :param runtime_filters: The filters provided at runtime, usually during a query operation execution. These filters
                            can change for each query/retriever run invocation.
    :param default_logical_operator: The default logical operator to use when merging filters (non-legacy filters only).
    :returns: A dictionary containing the resulting filters based on the provided policy.
    """
    if filter_policy == FilterPolicy.MERGE and runtime_filters and init_filters:
        # now we merge filters
        if is_comparison_filter(init_filters) and is_comparison_filter(runtime_filters):
            return combine_two_comparison_filters(init_filters, runtime_filters, default_logical_operator)
        elif is_comparison_filter(init_filters) and is_logical_filter(runtime_filters):
            return combine_init_comparison_and_runtime_logical_filters(
                init_filters, runtime_filters, default_logical_operator
            )
        elif is_logical_filter(init_filters) and is_comparison_filter(runtime_filters):
            return combine_runtime_comparison_and_init_logical_filters(
                runtime_filters, init_filters, default_logical_operator
            )
        elif is_logical_filter(init_filters) and is_logical_filter(runtime_filters):
            return combine_two_logical_filters(init_filters, runtime_filters)

    return runtime_filters or init_filters
