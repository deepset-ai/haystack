# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, Optional


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
        enum_map = {e.value: e for e in FilterPolicy}
        policy = enum_map.get(filter_policy)
        if policy is None:
            msg = f"Unknown FilterPolicy type '{filter_policy}'. Supported types are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return policy


def is_legacy(filter_item: Dict[str, Any]) -> bool:
    """
    Check if the given filter is a legacy filter.

    :param filter_item: The filter to check.
    :returns: True if the filter is a legacy filter, False otherwise.
    """
    return "operator" not in filter_item


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


def apply_filter_policy(
    filter_policy: FilterPolicy,
    init_filters: Optional[Dict[str, Any]] = None,
    runtime_filters: Optional[Dict[str, Any]] = None,
    default_logical_operator: str = "AND",
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
    if filter_policy == FilterPolicy.MERGE and runtime_filters:
        # legacy filters merge handling
        if is_legacy(runtime_filters):
            return {**(init_filters or {}), **runtime_filters}
        elif init_filters is not None:
            # here we merge new filters
            def merge_comparison_filters(
                filter1: Dict[str, Any], filter2: Dict[str, Any], logical_op: str
            ) -> Dict[str, Any]:
                if filter1["field"] == filter2["field"]:
                    # When fields are the same, use the runtime filter (filter2)
                    return filter2
                return {"operator": logical_op, "conditions": [filter1, filter2]}

            def merge_comparison_and_logical(
                comparison: Dict[str, Any], logical: Dict[str, Any], logical_op: str
            ) -> Dict[str, Any]:
                if logical["operator"] == logical_op:
                    logical["conditions"].append(comparison)
                    return logical
                else:
                    return {"operator": logical_op, "conditions": [comparison, logical]}

            def merge_logical_filters(
                filter1: Dict[str, Any], filter2: Dict[str, Any], logical_op: str
            ) -> Dict[str, Any]:
                if filter1["operator"] == filter2["operator"] == logical_op:
                    return {"operator": logical_op, "conditions": filter1["conditions"] + filter2["conditions"]}
                else:
                    return {"operator": logical_op, "conditions": [filter1, filter2]}

            if is_comparison_filter(init_filters) and is_comparison_filter(runtime_filters):
                return merge_comparison_filters(init_filters, runtime_filters, default_logical_operator)
            elif is_comparison_filter(init_filters) and is_logical_filter(runtime_filters):
                return merge_comparison_and_logical(init_filters, runtime_filters, default_logical_operator)
            elif is_logical_filter(init_filters) and is_comparison_filter(runtime_filters):
                return merge_comparison_and_logical(runtime_filters, init_filters, default_logical_operator)
            elif is_logical_filter(init_filters) and is_logical_filter(runtime_filters):
                return merge_logical_filters(init_filters, runtime_filters, default_logical_operator)

    return runtime_filters or init_filters
