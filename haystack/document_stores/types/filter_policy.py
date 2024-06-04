# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


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
