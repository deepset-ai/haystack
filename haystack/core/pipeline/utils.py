# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import heapq
from copy import deepcopy
from functools import wraps
from itertools import count
from typing import Any, List, Optional, Tuple

from haystack import logging
from haystack.core.component import Component
from haystack.tools import Tool, Toolset

logger = logging.getLogger(__name__)


def _deepcopy_with_exceptions(obj: Any) -> Any:
    """
    Attempts to perform a deep copy of the given object.

    This function recursively handles common container types (lists, tuples, sets, and dicts) to ensure deep copies
    of nested structures. For specific object types that are known to be problematic for deepcopying-such as
    instances of `Component`, `Tool`, or `Toolset` - the original object is returned as-is.
    If `deepcopy` fails for any other reason, the original object is returned and a log message is recorded.

    :param obj: The object to be deep-copied.

    :returns:
        A deep-copied version of the object, or the original object if deepcopying fails.
    """
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_deepcopy_with_exceptions(v) for v in obj)

    if isinstance(obj, dict):
        return {k: _deepcopy_with_exceptions(v) for k, v in obj.items()}

    # Components and Tools often contain objects that we do not want to deepcopy or are not deepcopyable
    # (e.g. models, clients, etc.). In this case we return the object as-is.
    if isinstance(obj, (Component, Tool, Toolset)):
        return obj

    try:
        return deepcopy(obj)
    except Exception as e:
        logger.info(
            "Deepcopy failed for object of type '{obj_type}'. Error: {error}. Returning original object instead.",
            obj_type=type(obj).__name__,
            error=e,
        )
        return obj


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


def args_deprecated(func):
    """
    Decorator to warn about the use of positional arguments in a function.

    Adapted from https://stackoverflow.com/questions/68432070/
    :param func:
    """

    def _positional_arg_warning() -> None:
        """
        Triggers a warning message if positional arguments are used in a function
        """
        import warnings

        msg = (
            "Warning: In an upcoming release, this method will require keyword arguments for all parameters. "
            "Please update your code to use keyword arguments to ensure future compatibility. "
            "Example: pipeline.draw(path='output.png', server_url='https://custom-server.com')"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # call the function first, to make sure the signature matches
        ret_value = func(*args, **kwargs)

        # A Pipeline instance is always the first argument - remove it from the args to check for positional arguments
        # We check the class name as strings to avoid circular imports
        if args and isinstance(args, tuple) and args[0].__class__.__name__ in ["Pipeline", "PipelineBase"]:
            args = args[1:]

        if args:
            _positional_arg_warning()
        return ret_value

    return wrapper
