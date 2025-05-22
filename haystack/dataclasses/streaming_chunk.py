# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from haystack.dataclasses.chat_message import ToolCallResult
from haystack.utils.asynchronous import is_callable_async_compatible


# Similar to ChoiceDeltaToolCall from OpenAI
@dataclass(kw_only=True)
class ToolCallDelta:
    """
    Represents a Tool call prepared by the model, usually contained in an assistant message.

    :param id: The ID of the Tool call.
    :param name: The name of the Tool to call.
    :param arguments:
    """

    index: int
    id: Optional[str] = None  # noqa: A003
    name: Optional[str] = None
    arguments: Optional[str] = None

    def __post_init__(self):
        if self.name is None and self.arguments is None:
            raise ValueError("At least one of tool_name or arguments must be provided.")
        if self.name is not None and self.arguments is not None:
            raise ValueError("Only one of tool_name or arguments can be provided.")


@dataclass
class StreamingChunk:
    """
    The StreamingChunk class encapsulates a segment of streamed content along with associated metadata.

    This structure facilitates the handling and processing of streamed data in a systematic manner.

    :param content: The content of the message chunk as a string.
    :param tool_calls: An optional ToolCallDelta object representing a tool call associated with the message chunk.
    :param tool_call_results: An optional ToolCallResult object representing the result of a tool call.
    :param start: A boolean indicating whether this chunk marks the start of a message.
    :param meta: A dictionary containing metadata related to the message chunk.
    """

    content: str
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    tool_calls: Optional[List[ToolCallDelta]] = None
    tool_call_results: Optional[List[ToolCallResult]] = None
    start: Optional[bool] = None

    def __post_init__(self):
        if self.tool_calls and self.content:
            raise ValueError("A StreamingChunk should not have both content and tool calls.")
        if self.tool_call_results and self.content:
            raise ValueError("A StreamingChunk should not have both content and tool call results.")
        if self.tool_calls and self.tool_call_results:
            raise ValueError("A StreamingChunk should not have both tool calls and tool call results.")


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
