# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haystack.dataclasses.streaming_chunk import StreamingCallbackT, StreamingChunk


def is_callable_async_compatible(func: Callable) -> bool:
    """
    Returns if the given callable is usable inside a component's `run_async` method.

    :param func:
        The callable to check.
    :returns:
        True if the callable is compatible, False otherwise.
    """
    return inspect.iscoroutinefunction(func)


async def _invoke_streaming_callback(callback: StreamingCallbackT, chunk: StreamingChunk) -> None:
    """
    Invokes a streaming callback in an async context, handling both sync and async callbacks.

    A sync callback runs inline and may block the event loop; an async callback is awaited.

    :param callback: The streaming callback to invoke.
    :param chunk: The streaming chunk to pass to the callback.
    """
    result = callback(chunk)
    if inspect.isawaitable(result):
        await result
