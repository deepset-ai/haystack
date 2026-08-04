# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any, TypeVar

from haystack import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _gather_tasks_with_cancel(tasks: list[asyncio.Task[T]]) -> list[T]:
    """
    Wait for all tasks, cancelling and draining unfinished siblings if one fails.

    :param tasks: Tasks to wait for.
    :returns:
        The task results in input order.
    """
    try:
        return await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _execute_component_async(component_instance: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Run a component asynchronously, preferring its `run_async` method when implemented.

    If the component does not implement `run_async`, its synchronous `run` method is executed in a thread
    to avoid blocking the event loop.

    :param component_instance: The component to run. Any object exposing a `run` method and optionally a
        `run_async` coroutine method.
    :param kwargs: Keyword arguments passed to the component's `run_async` or `run` method.
    :returns:
        The component's output dictionary.
    """
    run_async = getattr(component_instance, "run_async", None)
    if callable(run_async):
        return await run_async(**kwargs)

    logger.debug(
        "{component_type} does not implement 'run_async'. Running the synchronous 'run' method in a thread "
        "to avoid blocking the event loop.",
        component_type=type(component_instance).__name__,
    )
    return await asyncio.to_thread(component_instance.run, **kwargs)
