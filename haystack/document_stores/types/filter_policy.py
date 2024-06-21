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


def apply_filter_policy(
    filter_policy: FilterPolicy,
    init_filters: Optional[Dict[str, Any]] = None,
    runtime_filters: Optional[Dict[str, Any]] = None,
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
                            can change for each query/retreiver run invocation.
    :returns: A dictionary containing the resulting filters based on the provided policy.
    """
    if filter_policy == FilterPolicy.MERGE and runtime_filters:
        return {**(init_filters or {}), **runtime_filters}
    else:
        return runtime_filters or init_filters
