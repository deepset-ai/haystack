# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the StreamingChunk.

        :returns: Serialized dictionary representation of the calling object.
        """
        return {
            "content": self.content,
            "meta": self.meta,
            "component_info": asdict(self.component_info) if self.component_info else None,
            "index": self.index,
            "tool_calls": [asdict(tc) for tc in self.tool_calls] if self.tool_calls else None,
            "tool_call_result": asdict(self.tool_call_result) if self.tool_call_result else None,
            "start": self.start,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamingChunk":
        """
        Creates a deserialized StreamingChunk instance from a serialized representation.

        :param data: Dictionary containing the StreamingChunk's attributes.
        :returns: A StreamingChunk instance.
        """
        if "content" not in data:
            raise ValueError("Missing field 'content' in StreamingChunk deserialization. Field 'content' is required.")

        component_info = data.get("component_info")
        if isinstance(component_info, dict):
            component_info = ComponentInfo(**component_info)
        elif not (component_info is None or isinstance(component_info, ComponentInfo)):
            raise TypeError("component_info must be of type dict or ComponentInfo.")

        tool_calls = data.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise TypeError("tool_calls must be a list of ToolCallDelta.")
            checked_tool_calls = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    checked_tool_calls.append(ToolCallDelta(**tool_call))
                elif isinstance(tool_call, ToolCallDelta):
                    checked_tool_calls.append(tool_call)
                else:
                    raise TypeError("Each element of tool_calls must be of type dict or ToolCallDelta.")
            tool_calls = checked_tool_calls

        tool_call_result = data.get("tool_call_result")
        if isinstance(tool_call_result, dict):
            tool_call_result = ToolCallResult(**tool_call_result)
        elif not (tool_call_result is None or isinstance(tool_call_result, ToolCallResult)):
            raise TypeError("tool_call_result must be of type dict or ToolCallResult.")

        return StreamingChunk(
            content=data["content"],
            meta=data.get("meta", {}),
            component_info=component_info,
            index=data.get("index"),
            tool_calls=tool_calls,
            tool_call_result=tool_call_result,
            start=data.get("start", False),
            finish_reason=data.get("finish_reason"),
        )


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
