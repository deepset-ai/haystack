# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from haystack.utils.asynchronous import is_callable_async_compatible


class FinishReason(str, Enum):
    """
    Represents finish reasons for streaming chunks.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    TOOL_USE = "tool_use"
    ERROR = "error"


@dataclass
class ToolCall:
    """
    Represents a tool call request from the model.
    """

    id: str
    type: str
    function: Dict[str, Any]


@dataclass
class ToolResult:
    """
    Represents a result from a tool call.
    """

    tool_call_id: str
    name: str
    content: str


@dataclass
class Usage:
    """
    Represents token usage information.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class StreamingChunk:
    """
    The StreamingChunk class encapsulates a segment of streamed content along with associated metadata.

    This structure facilitates the handling and processing of streamed data in a systematic manner.

    :param content: The content of the message chunk as a string.
    :param meta: A dictionary containing metadata related to the message chunk.
    :param tool_calls: List of tool calls requested by the model.
    :param tool_results: List of results from tool calls.
    :param finish_reason: The reason why the model finished generating.
    :param usage: Token usage information.
    """

    content: str
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    finish_reason: Optional[FinishReason] = None
    usage: Optional[Usage] = None


SyncStreamingCallbackT = Callable[[StreamingChunk], None]
AsyncStreamingCallbackT = Callable[[StreamingChunk], Awaitable[None]]

StreamingCallbackT = Union[SyncStreamingCallbackT, AsyncStreamingCallbackT]


def select_streaming_callback(
    init_callback: Optional[StreamingCallbackT], runtime_callback: Optional[StreamingCallbackT], requires_async: bool
) -> Optional[StreamingCallbackT]:
    """
    Picks the correct streaming callback given an optional initial and runtime callback.

    The runtime callback takes precedence over the initial callback.

    :param init_callback:
        The initial callback.
    :param runtime_callback:
        The runtime callback.
    :param requires_async:
        Whether the selected callback must be async compatible.
    :returns:
        The selected callback.
    """
    if init_callback is not None:
        if requires_async and not is_callable_async_compatible(init_callback):
            raise ValueError("The init callback must be async compatible.")
        if not requires_async and is_callable_async_compatible(init_callback):
            raise ValueError("The init callback cannot be a coroutine.")

    if runtime_callback is not None:
        if requires_async and not is_callable_async_compatible(runtime_callback):
            raise ValueError("The runtime callback must be async compatible.")
        if not requires_async and is_callable_async_compatible(runtime_callback):
            raise ValueError("The runtime callback cannot be a coroutine.")

    return runtime_callback or init_callback
