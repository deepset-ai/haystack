# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import heapq

from itertools import count
from typing import Optional, Tuple


def parse_connect_string(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string.

    :param connection:
        The connection string.
    :returns:
        A tuple containing the component name and the connection name.
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


class FIFOPriorityQueue:
    """
    A priority queue that maintains FIFO order for items of equal priority.
    Items with the same priority are placed to the right of existing items.
    """

    def __init__(self):
        self._queue = []  # list of tuples (priority, count, item)
        self._counter = count()  # unique sequence of numbers

    def push(self, item, priority):
        """Push an item with given priority. Items with equal priority maintain FIFO order."""
        count = next(self._counter)
        entry = (priority, count, item)
        heapq.heappush(self._queue, entry)

    def pop(self):
        """Remove and return tuple of (priority, item) with lowest priority."""
        if not self._queue:
            raise IndexError("pop from empty queue")
        priority, count, item = heapq.heappop(self._queue)
        return priority, item

    def peek(self):
        """Return but don't remove tuple of (priority, item) with lowest priority."""
        if not self._queue:
            raise IndexError("peek at empty queue")
        priority, count, item = self._queue[0]
        return priority, item

    def get(self):
        """Remove and return tuple of (priority, item), or None if queue is empty."""
        if not self._queue:
            return None
        priority, count, item = heapq.heappop(self._queue)
        return priority, item

    def __len__(self):
        return len(self._queue)

    def __bool__(self):
        return bool(self._queue)
