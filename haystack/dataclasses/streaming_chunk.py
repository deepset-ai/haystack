# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, overload

from haystack.core.component import Component
from haystack.dataclasses.chat_message import ToolCallResult
from haystack.utils.asynchronous import is_callable_async_compatible

# Type alias for standard finish_reason values following OpenAI's convention
# plus Haystack-specific value ("tool_call_results")
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "tool_call_results"]


@dataclass
class ToolCallDelta:
    """
    Represents a Tool call prepared by the model, usually contained in an assistant message.

    :param index: The index of the Tool call in the list of Tool calls.
    :param tool_name: The name of the Tool to call.
    :param arguments: Either the full arguments in JSON format or a delta of the arguments.
    :param id: The ID of the Tool call.
    """

    index: int
    tool_name: Optional[str] = field(default=None)
    arguments: Optional[str] = field(default=None)
    id: Optional[str] = field(default=None)  # noqa: A003

    def __post_init__(self):
        # NOTE: We allow for name and arguments to both be present because some providers like Mistral provide the
        # name and full arguments in one chunk
        if self.tool_name is None and self.arguments is None:
            raise ValueError("At least one of tool_name or arguments must be provided.")


@dataclass
class ComponentInfo:
    """
    The `ComponentInfo` class encapsulates information about a component.

    :param type: The type of the component.
    :param name: The name of the component assigned when adding it to a pipeline.

    """

    type: str
    name: Optional[str] = field(default=None)

    @classmethod
    def from_component(cls, component: Component) -> "ComponentInfo":
        """
        Create a `ComponentInfo` object from a `Component` instance.

        :param component:
            The `Component` instance.
        :returns:
            The `ComponentInfo` object with the type and name of the given component.
        """
        component_type = f"{component.__class__.__module__}.{component.__class__.__name__}"
        component_name = getattr(component, "__component_name__", None)
        return cls(type=component_type, name=component_name)


@dataclass
class StreamingChunk:
    """
    The `StreamingChunk` class encapsulates a segment of streamed content along with associated metadata.

    This structure facilitates the handling and processing of streamed data in a systematic manner.

    :param content: The content of the message chunk as a string.
    :param meta: A dictionary containing metadata related to the message chunk.
    :param component_info: A `ComponentInfo` object containing information about the component that generated the chunk,
        such as the component name and type.
    :param index: An optional integer index representing which content block this chunk belongs to.
    :param tool_calls: An optional list of ToolCallDelta object representing a tool call associated with the message
        chunk.
    :param tool_call_result: An optional ToolCallResult object representing the result of a tool call.
    :param start: A boolean indicating whether this chunk marks the start of a content block.
    :param finish_reason: An optional value indicating the reason the generation finished.
        Standard values follow OpenAI's convention: "stop", "length", "tool_calls", "content_filter",
        plus Haystack-specific value "tool_call_results".
    """

    content: str
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    component_info: Optional[ComponentInfo] = field(default=None)
    index: Optional[int] = field(default=None)
    tool_calls: Optional[List[ToolCallDelta]] = field(default=None)
    tool_call_result: Optional[ToolCallResult] = field(default=None)
    start: bool = field(default=False)
    finish_reason: Optional[FinishReason] = field(default=None)

    def __post_init__(self):
        fields_set = sum(bool(x) for x in (self.content, self.tool_calls, self.tool_call_result))
        if fields_set > 1:
            raise ValueError(
                "Only one of `content`, `tool_call`, or `tool_call_result` may be set in a StreamingChunk. "
                f"Got content: '{self.content}', tool_call: '{self.tool_calls}', "
                f"tool_call_result: '{self.tool_call_result}'"
            )

        # NOTE: We don't enforce this for self.content otherwise it would be a breaking change
        if (self.tool_calls or self.tool_call_result) and self.index is None:
            raise ValueError("If `tool_call`, or `tool_call_result` is set, `index` must also be set.")


SyncStreamingCallbackT = Callable[[StreamingChunk], None]
AsyncStreamingCallbackT = Callable[[StreamingChunk], Awaitable[None]]

StreamingCallbackT = Union[SyncStreamingCallbackT, AsyncStreamingCallbackT]


@overload
def select_streaming_callback(
    init_callback: Optional[StreamingCallbackT],
    runtime_callback: Optional[StreamingCallbackT],
    requires_async: Literal[False],
) -> Optional[SyncStreamingCallbackT]: ...
@overload
def select_streaming_callback(
    init_callback: Optional[StreamingCallbackT],
    runtime_callback: Optional[StreamingCallbackT],
    requires_async: Literal[True],
) -> Optional[AsyncStreamingCallbackT]: ...


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
