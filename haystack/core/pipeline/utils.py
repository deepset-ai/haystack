# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
from itertools import count
from typing import Any, List, Optional, Tuple


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

    Items with the same priority are processed in the order they were added.
    This queue ensures that when multiple items share the same priority level,
    they are dequeued in the same order they were enqueued (First-In-First-Out).
    """

    def __init__(self) -> None:
        """
        Initialize a new FIFO priority queue.
        """
        # List of tuples (priority, count, item) where count ensures FIFO order
        self._queue: List[Tuple[int, int, Any]] = []
        # Counter to maintain insertion order for equal priorities
        self._counter = count()

    def push(self, item: Any, priority: int) -> None:
        """
        Push an item into the queue with a given priority.

        Items with equal priority maintain FIFO ordering based on insertion time.
        Lower priority numbers are dequeued first.

        :param item:
            The item to insert into the queue.
        :param priority:
            Priority level for the item. Lower numbers indicate higher priority.
        """
        next_count = next(self._counter)
        entry = (priority, next_count, item)
        heapq.heappush(self._queue, entry)

    def pop(self) -> Tuple[int, Any]:
        """
        Remove and return the highest priority item from the queue.

        For items with equal priority, returns the one that was inserted first.

        :returns:
            A tuple containing (priority, item) with the lowest priority number.
        :raises IndexError:
            If the queue is empty.
        """
        if not self._queue:
            raise IndexError("pop from empty queue")
        priority, _, item = heapq.heappop(self._queue)
        return priority, item

    def peek(self) -> Tuple[int, Any]:
        """
        Return but don't remove the highest priority item from the queue.

        For items with equal priority, returns the one that was inserted first.

        :returns:
            A tuple containing (priority, item) with the lowest priority number.
        :raises IndexError:
            If the queue is empty.
        """
        if not self._queue:
            raise IndexError("peek at empty queue")
        priority, _, item = self._queue[0]
        return priority, item

    def get(self) -> Optional[Tuple[int, Any]]:
        """
        Remove and return the highest priority item from the queue.

        For items with equal priority, returns the one that was inserted first.
        Unlike pop(), returns None if the queue is empty instead of raising an exception.

        :returns:
            A tuple containing (priority, item), or None if the queue is empty.
        """
        if not self._queue:
            return None
        priority, _, item = heapq.heappop(self._queue)
        return priority, item

    def __len__(self) -> int:
        """
        Return the number of items in the queue.

        :returns:
            The number of items currently in the queue.
        """
        return len(self._queue)

    def __bool__(self) -> bool:
        """
        Return True if the queue has items, False if empty.

        :returns:
            True if the queue contains items, False otherwise.
        """
        return bool(self._queue)
