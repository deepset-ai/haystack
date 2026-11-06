# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import reprlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from haystack import logging
from haystack.dataclasses.file_content import FileContent
from haystack.dataclasses.image_content import ImageContent
from haystack.utils.dataclasses import _warn_on_inplace_mutation

logger = logging.getLogger(__name__)

# Shows OpenAI content parts in error messages without their base64 payloads. The string limit matches the reprs of
# ImageContent and FileContent.
_CONTENT_PART_REPR = reprlib.Repr()
_CONTENT_PART_REPR.maxstring = 100


class ChatRole(str, Enum):  # noqa: UP042 # StrEnum would change str() of this public enum
    """
    Enumeration representing the roles within a chat.
    """

    #: The user role. A message from the user can contain text, images, and files.
    USER = "user"

    #: The system role. A message from the system contains only text.
    SYSTEM = "system"

    #: The assistant role. A message from the assistant can contain text and Tool calls. It can also store metadata.
    ASSISTANT = "assistant"

    #: The tool role. A message from a tool contains the result of a Tool invocation.
    TOOL = "tool"

    @staticmethod
    def from_str(string: str) -> "ChatRole":
        """
        Convert a string to a ChatRole enum.
        """
        enum_map = {e.value: e for e in ChatRole}
        role = enum_map.get(string)
        if role is None:
            msg = f"Unknown chat role '{string}'. Supported roles are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return role


@_warn_on_inplace_mutation
@dataclass
class TextContent:
    """
    The textual content of a chat message.

    :param text: The text content of the message.
    """

    text: str

    def to_dict(self) -> dict[str, Any]:
        """
        Convert TextContent into a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextContent":
        """
        Create a TextContent from a dictionary.
        """
        return TextContent(**data)


@_warn_on_inplace_mutation
@dataclass
class ToolCall:
    """
    Represents a Tool call prepared by the model, usually contained in an assistant message.

    :param id: The ID of the Tool call.
    :param tool_name: The name of the Tool to call.
    :param arguments: The arguments to call the Tool with.
    :param extra: Dictionary of extra information about the Tool call. Use to store provider-specific
        information. To avoid serialization issues, values should be JSON serializable.
    """

    tool_name: str
    arguments: dict[str, Any]
    id: str | None = None  # noqa: A003
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ToolCall into a dictionary.

        :returns: A dictionary with keys 'tool_name', 'arguments', 'id', and 'extra'.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """
        Creates a new ToolCall object from a dictionary.

        :param data:
            The dictionary to build the ToolCall object.
        :returns:
            The created object.
        """
        return ToolCall(**data)


ToolCallResultContentT = str | Sequence[TextContent | ImageContent | FileContent]


@_warn_on_inplace_mutation
@dataclass
class ToolCallResult:
    """
    Represents the result of a Tool invocation.

    :param result: The result of the Tool invocation.
    :param origin: The Tool call that produced this result.
    :param error: Whether the Tool invocation resulted in an error.
    """

    result: ToolCallResultContentT
    origin: ToolCall
    error: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Converts ToolCallResult into a dictionary.

        :returns: A dictionary with keys 'result', 'origin', and 'error'.
        """
        return self._to_dict()

    def _to_trace_dict(self) -> dict[str, Any]:
        """
        Convert ToolCallResult into a dictionary for tracing, redacting large binary payloads.

        :returns:
            Serialized version of the object only for tracing purposes.
        """
        return self._to_dict(for_trace=True)

    def _to_dict(self, *, for_trace: bool = False) -> dict[str, Any]:
        """Serialize the tool result, optionally replacing binary payloads with tracing placeholders."""
        result: Any = self.result
        if not isinstance(result, str):
            if not isinstance(result, Sequence) or not all(
                isinstance(part, (TextContent, ImageContent, FileContent)) for part in result
            ):
                raise ValueError(
                    "ToolCallResult result must be a string or a sequence of TextContent, ImageContent, or FileContent"
                )
            result = [_serialize_content_part(part, for_trace=for_trace) for part in result]
        return {"result": result, "origin": self.origin.to_dict(), "error": self.error}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallResult":
        """
        Creates a ToolCallResult from a dictionary.

        :param data:
            The dictionary to build the ToolCallResult object.
        :returns:
            The created object.
        """
        if not all(x in data for x in ["result", "origin", "error"]):
            raise ValueError(
                "Fields `result`, `origin`, `error` are required for ToolCallResult deserialization. "
                f"Received dictionary with keys {list(data.keys())}"
            )

        result = data["result"]
        if isinstance(result, list):
            result = [_deserialize_content_part(serialized_part) for serialized_part in result]

        return ToolCallResult(result=result, origin=ToolCall.from_dict(data["origin"]), error=data["error"])


@_warn_on_inplace_mutation
@dataclass
class ReasoningContent:
    """
    Represents the optional reasoning content prepared by the model, usually contained in an assistant message.

    :param reasoning_text: The reasoning text produced by the model.
    :param extra: Dictionary of extra information about the reasoning content. Use to store provider-specific
        information. To avoid serialization issues, values should be JSON serializable.
    """

    reasoning_text: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ReasoningContent into a dictionary.

        :returns: A dictionary with keys 'reasoning_text', and 'extra'.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningContent":
        """
        Creates a new ReasoningContent object from a dictionary.

        :param data:
            The dictionary to build the ReasoningContent object.
        :returns:
            The created object.
        """
        return ReasoningContent(**data)


ChatMessageContentT = TextContent | ToolCall | ToolCallResult | ImageContent | ReasoningContent | FileContent


_CONTENT_PART_CLASSES_TO_SERIALIZATION_KEYS: dict[type[ChatMessageContentT], str] = {
    TextContent: "text",
    ToolCall: "tool_call",
    ToolCallResult: "tool_call_result",
    ImageContent: "image",
    ReasoningContent: "reasoning",
    FileContent: "file",
}


def _serialize_content_part(part: ChatMessageContentT, *, for_trace: bool = False) -> dict[str, Any]:
    """
    Serialize a single content part of a ChatMessage.

    :param part:
        A ChatMessageContentT object.
    :param for_trace:
        Whether to redact large binary payloads for tracing.
    :returns:
        A dictionary representing the content part.
    :raises TypeError:
        If the part is not a valid ChatMessageContentT object.
    """
    serialization_key = _CONTENT_PART_CLASSES_TO_SERIALIZATION_KEYS.get(type(part))
    if serialization_key is None:
        raise TypeError(f"Unsupported type in ChatMessage content: `{type(part).__name__}` for `{part}`.")

    data = part._to_trace_dict() if for_trace and hasattr(part, "_to_trace_dict") else part.to_dict()
    # Text is serialized as {"text": "..."}; other parts are nested, e.g. {"image": {"base64_image": "...", ...}}.
    return data if isinstance(part, TextContent) else {serialization_key: data}


def _deserialize_content_part(serialized_part: dict[str, Any]) -> ChatMessageContentT:
    """
    Deserialize a single content part of a serialized ChatMessage.

    :param serialized_part:
        A dictionary representing a single content part of a serialized ChatMessage.
    :returns:
        A ChatMessageContentT object.
    :raises ValueError:
        If the dictionary does not represent a supported content part.
    """
    # handle flat text format separately
    if "text" in serialized_part:
        return TextContent.from_dict(serialized_part)

    for cls, serialization_key in _CONTENT_PART_CLASSES_TO_SERIALIZATION_KEYS.items():
        if serialization_key in serialized_part:
            return cls.from_dict(serialized_part[serialization_key])

    # Support for Pydantic's model_dump() output, which produces a flat dictionary without wrapping keys.
    if "tool_name" in serialized_part and "arguments" in serialized_part:
        return ToolCall.from_dict(serialized_part)
    if "result" in serialized_part and "origin" in serialized_part:
        return ToolCallResult.from_dict(serialized_part)
    if "reasoning_text" in serialized_part:
        return ReasoningContent.from_dict(serialized_part)
    if "base64_image" in serialized_part:
        return ImageContent.from_dict(serialized_part)
    if "base64_data" in serialized_part:
        return FileContent.from_dict(serialized_part)

    # NOTE: this verbose error message provides guidance to LLMs when creating invalid messages during agent runs
    msg = (
        f"Unsupported content part in the serialized ChatMessage: {serialized_part}. "
        "The `content` field of the serialized ChatMessage must be a list of dictionaries, where each dictionary "
        "contains one of these keys: 'text', 'image', 'file', 'reasoning', 'tool_call', or 'tool_call_result'. "
        "Valid formats: [{'text': 'Hello'}, {'image': {'base64_image': '...', ...}}, "
        "{'file': {'base64_data': '...', ...}}, {'reasoning': {'reasoning_text': 'I think...', 'extra': {...}}}, "
        "{'tool_call': {'tool_name': 'search', 'arguments': {}, 'id': 'call_123'}}, "
        "{'tool_call_result': {'result': 'data', 'origin': {...}, 'error': false}}]"
    )
    raise ValueError(msg)


@_warn_on_inplace_mutation
@dataclass
class ChatMessage:
    """
    Represents a message in a LLM chat conversation.

    Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a ChatMessage.
    """

    _role: ChatRole
    _content: Sequence[ChatMessageContentT]
    _name: str | None = None
    _meta: dict[str, Any] = field(default_factory=dict, hash=False)

    def __len__(self) -> int:
        return len(self._content)

    @property
    def role(self) -> ChatRole:
        """
        Returns the role of the entity sending the message.
        """
        return self._role

    @property
    def meta(self) -> dict[str, Any]:
        """
        Returns the metadata associated with the message.
        """
        return self._meta

    @property
    def name(self) -> str | None:
        """
        Returns the name associated with the message.
        """
        return self._name

    @property
    def texts(self) -> list[str]:
        """
        Returns the list of all texts contained in the message.
        """
        return [content.text for content in self._content if isinstance(content, TextContent)]

    @property
    def text(self) -> str | None:
        """
        Returns the first text contained in the message.
        """
        if texts := self.texts:
            return texts[0]
        return None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """
        Returns the list of all Tool calls contained in the message.
        """
        return [content for content in self._content if isinstance(content, ToolCall)]

    @property
    def tool_call(self) -> ToolCall | None:
        """
        Returns the first Tool call contained in the message.
        """
        if tool_calls := self.tool_calls:
            return tool_calls[0]
        return None

    @property
    def tool_call_results(self) -> list[ToolCallResult]:
        """
        Returns the list of all Tool call results contained in the message.
        """
        return [content for content in self._content if isinstance(content, ToolCallResult)]

    @property
    def tool_call_result(self) -> ToolCallResult | None:
        """
        Returns the first Tool call result contained in the message.
        """
        if tool_call_results := self.tool_call_results:
            return tool_call_results[0]
        return None

    @property
    def images(self) -> list[ImageContent]:
        """
        Returns the list of all images contained in the message.
        """
        return [content for content in self._content if isinstance(content, ImageContent)]

    @property
    def image(self) -> ImageContent | None:
        """
        Returns the first image contained in the message.
        """
        if images := self.images:
            return images[0]
        return None

    @property
    def files(self) -> list[FileContent]:
        """
        Returns the list of all files contained in the message.
        """
        return [content for content in self._content if isinstance(content, FileContent)]

    @property
    def file(self) -> FileContent | None:
        """
        Returns the first file contained in the message.
        """
        if files := self.files:
            return files[0]
        return None

    @property
    def reasonings(self) -> list[ReasoningContent]:
        """
        Returns the list of all reasoning contents contained in the message.
        """
        return [content for content in self._content if isinstance(content, ReasoningContent)]

    @property
    def reasoning(self) -> ReasoningContent | None:
        """
        Returns the first reasoning content contained in the message.
        """
        if reasonings := self.reasonings:
            return reasonings[0]
        return None

    def is_from(self, role: ChatRole | str) -> bool:
        """
        Check if the message is from a specific role.

        :param role: The role to check against.
        :returns: True if the message is from the specified role, False otherwise.
        """
        if isinstance(role, str):
            role = ChatRole.from_str(role)
        return self._role == role

    @classmethod
    def from_user(
        cls,
        text: str | None = None,
        meta: dict[str, Any] | None = None,
        name: str | None = None,
        *,
        content_parts: Sequence[TextContent | str | ImageContent | FileContent] | None = None,
    ) -> "ChatMessage":
        """
        Create a message from the user.

        :param text: The text content of the message. Specify this or content_parts.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :param content_parts: A list of content parts to include in the message. Specify this or text.
        :returns: A new ChatMessage instance.
        :raises ValueError: If neither or both of text and content_parts are provided, or if content_parts is empty.
        :raises TypeError: If a content part is not a str, TextContent, ImageContent, or FileContent.
        """
        if text is None and content_parts is None:
            raise ValueError("Either text or content_parts must be provided.")
        if text is not None and content_parts is not None:
            raise ValueError("Only one of text or content_parts can be provided.")

        content: list[TextContent | ImageContent | FileContent] = []

        if text is not None:
            content = [TextContent(text=text)]
        elif content_parts is not None:
            for part in content_parts:
                if isinstance(part, str):
                    content.append(TextContent(text=part))
                elif isinstance(part, (TextContent, ImageContent, FileContent)):
                    content.append(part)
                else:
                    raise TypeError(
                        f"The user message must contain only text, image, or file parts. Unsupported part: {part}"
                    )
            if len(content) == 0:
                raise ValueError("The user message must contain at least one content part (text, image, file).")

        return cls(_role=ChatRole.USER, _content=content, _meta=meta or {}, _name=name)

    @classmethod
    def from_system(cls, text: str, meta: dict[str, Any] | None = None, name: str | None = None) -> "ChatMessage":
        """
        Create a message from the system.

        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :returns: A new ChatMessage instance.
        """
        return cls(_role=ChatRole.SYSTEM, _content=[TextContent(text=text)], _meta=meta or {}, _name=name)

    @classmethod
    def from_assistant(
        cls,
        text: str | None = None,
        meta: dict[str, Any] | None = None,
        name: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        *,
        reasoning: str | ReasoningContent | None = None,
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :param tool_calls: The Tool calls to include in the message.
        :param reasoning: The reasoning content to include in the message.
        :returns: A new ChatMessage instance.
        :raises TypeError: If `reasoning` is not a string or ReasoningContent object.
        """
        content: list[ChatMessageContentT] = []
        if reasoning:
            if isinstance(reasoning, str):
                content.append(ReasoningContent(reasoning_text=reasoning))
            elif isinstance(reasoning, ReasoningContent):
                content.append(reasoning)
            else:
                raise TypeError(f"reasoning must be a string or a ReasoningContent object, got {type(reasoning)}")
        if text is not None:
            content.append(TextContent(text=text))
        if tool_calls:
            content.extend(tool_calls)

        return cls(_role=ChatRole.ASSISTANT, _content=content, _meta=meta or {}, _name=name)

    @classmethod
    def from_tool(
        cls,
        tool_result: ToolCallResultContentT,
        origin: ToolCall,
        error: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> "ChatMessage":
        """
        Create a message from a Tool.

        :param tool_result: The result of the Tool invocation.
        :param origin: The Tool call that produced this result.
        :param error: Whether the Tool invocation resulted in an error.
        :param meta: Additional metadata associated with the message.
        :returns: A new ChatMessage instance.
        """
        return cls(
            _role=ChatRole.TOOL,
            _content=[ToolCallResult(result=tool_result, origin=origin, error=error)],
            _meta=meta or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """

        return self._to_dict()

    def _to_trace_dict(self) -> dict[str, Any]:
        """
        Convert the ChatMessage to a dictionary representation for tracing.

        Image and file payloads, including those in tool results, are replaced with placeholder strings to avoid
        sending large payloads to the tracing backend.

        :returns:
            Serialized version of the object only for tracing purposes.
        """
        return self._to_dict(for_trace=True)

    def _to_dict(self, *, for_trace: bool = False) -> dict[str, Any]:
        """Serialize the message, optionally redacting binary payloads in its content and tool results."""
        return {
            "role": self._role.value,
            "meta": self._meta,
            "name": self._name,
            "content": [_serialize_content_part(part, for_trace=for_trace) for part in self._content],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        """
        Creates a new ChatMessage object from a dictionary.

        :param data:
            The dictionary to build the ChatMessage object.
        :returns:
            The created object.
        :raises ValueError: If the role or content is missing, or the role is invalid.
        :raises TypeError: If the content is not a string or a sequence of content-part dictionaries.
        """

        # Pydantic dataclass dumps and pre-2.12 messages use underscore-prefixed fields.
        if "content" not in data and "_content" in data:
            fields = ("role", "content", "name", "meta")
            data = {key: data[f"_{key}"] for key in fields if f"_{key}" in data}

        # NOTE: this verbose error message provides guidance to LLMs when creating invalid messages during agent runs
        if "role" not in data:
            raise ValueError(
                "The `role` field is required in the message dictionary. "
                f"Expected a dictionary with 'role' field containing one of: {[role.value for role in ChatRole]}. "
                f"Common roles are 'user' (for user messages) and 'assistant' (for AI responses). "
                f"Received dictionary with keys: {list(data.keys())}"
            )
        if "content" not in data:
            raise ValueError(f"Missing 'content' or '_content' in serialized ChatMessage: `{data}`")

        serialized_content = data["content"]
        if isinstance(serialized_content, str):
            content: list[ChatMessageContentT] = [TextContent(text=serialized_content)]  # pre-2.9 format
        elif isinstance(serialized_content, Sequence):
            content = [_deserialize_content_part(serialized_part) for serialized_part in serialized_content]
        else:
            raise TypeError(f"Unsupported content type in serialized ChatMessage: `{serialized_content}`")

        return cls(_role=ChatRole(data["role"]), _content=content, _name=data.get("name"), _meta=data.get("meta") or {})

    def to_openai_dict_format(self, require_tool_call_ids: bool = True) -> dict[str, Any]:
        """
        Convert a ChatMessage to the dictionary format expected by OpenAI's Chat Completions API.

        The `_meta` field of ChatMessage is removed because it is not supported by OpenAI's Chat Completions API.
        The exception is a non-empty string in `meta["refusal"]` of an assistant message. It is sent as a refusal
        content part, or as the `refusal` field if the message also has text. If the message also has tool calls, the
        refusal is dropped with a warning, because OpenAI ignores the tool calls of a message with a refusal.
        A tool result can be a string or a sequence of `TextContent` parts, including a tuple.

        :param require_tool_call_ids:
            If True (default), enforces that each Tool Call includes a non-null `id` attribute.
            Set to False to allow Tool Calls without `id`, which may be suitable for shallow OpenAI-compatible APIs.
        :returns:
            The ChatMessage in the format expected by OpenAI's Chat Completions API.

        :raises ValueError:
            If the message format is invalid, or if `require_tool_call_ids` is True and any Tool Call is missing an
            `id` attribute.
        """

        has_content = bool(self.texts or self.tool_calls or self.tool_call_results or self.images or self.files)
        # We convert an assistant message with no content part into a message with empty content, which the API accepts
        if not has_content and not self.is_from(ChatRole.ASSISTANT):
            raise ValueError(
                f"A `ChatMessage` from `{self._role.value}` must contain at least one `TextContent`, `ToolCall`, "
                "`ToolCallResult`, `ImageContent`, or `FileContent`. Only assistant messages can be empty."
            )
        if len(self.tool_call_results) > 0 and len(self._content) > 1:
            raise ValueError(
                "For OpenAI compatibility, a `ChatMessage` with a `ToolCallResult` cannot contain any other content."
            )

        if self.tool_calls and not self.is_from(ChatRole.ASSISTANT):
            raise ValueError("Tool calls are only supported in assistant messages.")

        openai_msg: dict[str, Any] = {"role": self._role.value}
        if self._name is not None:
            openai_msg["name"] = self._name

        if self.is_from(ChatRole.USER):
            openai_msg.update(self._build_openai_user_message_fields())
        elif self.tool_call_results:
            openai_msg.update(
                self._build_openai_tool_result_message_fields(require_tool_call_ids=require_tool_call_ids)
            )
        elif self.is_from(ChatRole.ASSISTANT):
            openai_msg.update(self._build_openai_assistant_message_fields(require_tool_call_ids=require_tool_call_ids))
        else:
            openai_msg["content"] = self.text or ""
        return openai_msg

    def _build_openai_user_message_fields(self) -> dict[str, Any]:
        """Build the OpenAI-specific fields for a user message."""
        if len(self._content) == 1 and isinstance(self._content[0], TextContent):
            return {"content": self.text}
        return {
            "content": [
                _haystack_content_part_to_openai_dict(haystack_part)
                for haystack_part in self._content
                if isinstance(haystack_part, (TextContent, ImageContent, FileContent))
            ]
        }

    def _build_openai_tool_result_message_fields(self, require_tool_call_ids: bool) -> dict[str, Any]:
        """Build the OpenAI-specific fields for a tool result message."""
        tool_call_result = self.tool_call_results[0]
        openai_fields: dict[str, Any] = {}
        if isinstance(tool_call_result.result, str):
            openai_fields["content"] = tool_call_result.result
        # OpenAI Chat Completions API does not support multimodal tool results
        elif isinstance(tool_call_result.result, Sequence) and all(
            isinstance(haystack_part, TextContent) for haystack_part in tool_call_result.result
        ):
            openai_fields["content"] = [
                {"type": "text", "text": haystack_part.text}
                for haystack_part in tool_call_result.result
                if isinstance(haystack_part, TextContent)
            ]
        else:
            raise ValueError(
                f"Unsupported tool result: {tool_call_result}. If you need to pass images in tool results, "
                "use OpenAI Responses API instead."
            )

        if tool_call_result.origin.id is not None:
            openai_fields["tool_call_id"] = tool_call_result.origin.id
        elif require_tool_call_ids:
            raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
        # OpenAI does not provide a way to communicate errors in tool invocations, so we ignore the error field
        return openai_fields

    def _build_openai_assistant_message_fields(self, require_tool_call_ids: bool) -> dict[str, Any]:
        """Build the OpenAI-specific fields for an assistant message."""
        openai_fields: dict[str, Any] = {}
        text = self.text
        tool_calls = self.tool_calls
        # OpenAI Chat Completions API does not support reasoning content, so we ignore it
        if text is not None:
            openai_fields["content"] = text
        if tool_calls:
            if require_tool_call_ids and any(haystack_tool_call.id is None for haystack_tool_call in tool_calls):
                raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
            openai_fields["tool_calls"] = [
                _haystack_tool_call_to_openai_dict(haystack_tool_call) for haystack_tool_call in tool_calls
            ]
        refusal = self._meta.get("refusal")
        # OpenAI's content array holds either text parts or exactly one refusal part, so a refusal next to text goes
        # into the `refusal` field.
        if isinstance(refusal, str) and refusal:
            if tool_calls:
                logger.warning(
                    "Dropping the refusal of an assistant message that also has tool calls, because OpenAI ignores the "
                    "tool calls of a message with a refusal and then rejects the tool results that follow."
                )
            elif text is not None:
                openai_fields["refusal"] = refusal
            else:
                openai_fields["content"] = [{"type": "refusal", "refusal": refusal}]
        # The API rejects a message carrying neither content nor tool calls, so send empty content for it.
        if "content" not in openai_fields and "tool_calls" not in openai_fields:
            openai_fields["content"] = ""
        return openai_fields

    @classmethod
    def from_openai_dict_format(cls, message: dict[str, Any]) -> "ChatMessage":
        """
        Create a ChatMessage from a dictionary in the format expected by OpenAI's Chat Completions API.

        `content` can be a string or a list of content parts. In user messages, `text` parts become `TextContent`,
        `image_url` parts with a base64 data URL become `ImageContent`, and `file` parts with inline `file_data` become
        `FileContent`. System, developer, and assistant messages accept `text` parts, which are joined with a newline
        into a single text. An assistant's refusal, given as a `refusal` part or as the `refusal` field, is stored in
        `meta["refusal"]` instead of the text. Assistant messages with tool calls or a refusal can omit `content`
        or set it to None.

        NOTE: This method accepts tool calls without `id` and tool messages without `tool_call_id` to support
        shallow OpenAI-compatible APIs. Include these IDs when using the resulting ChatMessage with
        OpenAI Chat Completions to avoid validation errors.

        :param message:
            The OpenAI dictionary to build the ChatMessage object.
        :returns:
            The created ChatMessage object.

        :raises ValueError:
            If the message dictionary is missing required fields, if non-null `content` is neither a string nor a
            list, or if it contains content parts that can't be converted, such as image URLs that are not base64
            data URLs or files referenced by `file_id`.
        """

        if "role" not in message:
            raise ValueError("The `role` field is required in the message dictionary.")
        role = message["role"]
        if role not in ["assistant", "user", "system", "developer", "tool"]:
            raise ValueError(f"Unsupported role: {role}")
        openai_content = message.get("content")
        name = message.get("name")
        if openai_content is not None and not isinstance(openai_content, (str, list)):
            raise ValueError(
                f"The `content` field must be a string or a list of content parts, got {type(openai_content).__name__}."
            )

        if role == "assistant":
            refusal = message.get("refusal")
            openai_tool_calls = message.get("tool_calls")
            if refusal is not None and not isinstance(refusal, str):
                raise ValueError(f"The `refusal` field must be a string, got {type(refusal).__name__}.")
            # An empty string is valid content for an assistant message: that is how a reply with nothing to send
            # is serialized. Other falsy content requires tool calls or a refusal.
            if (openai_content is None or openai_content == []) and not openai_tool_calls and not refusal:
                raise ValueError(
                    "For assistant messages, either `content`, `tool_calls`, or `refusal` must be present."
                )
            parsed_content = _parse_openai_text_and_refusal_content(openai_content, role=role)
            refusals = [refusal] if refusal else []
            if "refusal" in parsed_content:
                refusals.append(parsed_content["refusal"])
            # Avoid duplicating a refusal present in both the field and the content parts.
            meta = {"refusal": "\n".join(dict.fromkeys(refusals))} if refusals else None
            return cls.from_assistant(
                text=parsed_content.get("text"),
                name=name,
                meta=meta,
                tool_calls=[
                    _openai_dict_to_haystack_tool_call(openai_tool_call) for openai_tool_call in openai_tool_calls or []
                ],
            )

        if openai_content is None or openai_content == []:
            raise ValueError(f"The `content` field is required for {role} messages.")
        if role == "user":
            if isinstance(openai_content, str):
                return cls.from_user(text=openai_content, name=name)
            return cls.from_user(
                content_parts=[_openai_dict_to_haystack_content_part(openai_part) for openai_part in openai_content],
                name=name,
            )
        if role == "tool":
            if isinstance(openai_content, list):
                if not all(isinstance(openai_part, dict) and "text" in openai_part for openai_part in openai_content):
                    raise ValueError(
                        "To be used with OpenAI Chat Completions, tool results must be a string "
                        "or a list of TextContent"
                    )
                tool_result: str | list[TextContent] = [
                    TextContent(text=openai_part["text"]) for openai_part in openai_content
                ]
            else:
                tool_result = openai_content
            return cls.from_tool(
                tool_result=tool_result, origin=ToolCall(id=message.get("tool_call_id"), tool_name="", arguments={})
            )

        text = _parse_openai_text_and_refusal_content(openai_content, role=role)["text"]
        return cls.from_system(text=text, name=name)


def _haystack_content_part_to_openai_dict(haystack_part: TextContent | ImageContent | FileContent) -> dict[str, Any]:
    """Convert a text, image, or file part to OpenAI Chat Completions format."""
    if isinstance(haystack_part, TextContent):
        return {"type": "text", "text": haystack_part.text}
    if isinstance(haystack_part, ImageContent):
        # If no MIME type is provided, default to JPEG.
        # OpenAI API appears to tolerate MIME type mismatches.
        image_url = {"url": f"data:{haystack_part.mime_type or 'image/jpeg'};base64,{haystack_part.base64_image}"}
        if haystack_part.detail:
            image_url["detail"] = haystack_part.detail
        return {"type": "image_url", "image_url": image_url}
    return {
        "type": "file",
        "file": {
            "file_data": f"data:{haystack_part.mime_type or 'application/pdf'};base64,{haystack_part.base64_data}",
            # Filename is optional but if not provided, OpenAI expects a file_id of a previous file upload.
            # We use a dummy filename.
            "filename": haystack_part.filename or "filename",
        },
    }


def _openai_dict_to_haystack_content_part(openai_part: Any) -> TextContent | ImageContent | FileContent:
    """
    Convert a single content part in OpenAI format into a Haystack content part.

    :param openai_part: A content part in OpenAI format.
    :returns: A TextContent, ImageContent, or FileContent object.
    :raises ValueError: If a content part is malformed or of an unsupported type.
    """
    part_type = openai_part.get("type") if isinstance(openai_part, dict) else None
    if part_type == "text":
        if not isinstance(openai_part.get("text"), str):
            raise ValueError(
                f"Unsupported text content part: {_CONTENT_PART_REPR.repr(openai_part)}. "
                "Text parts must contain a `text` string."
            )
        return TextContent(text=openai_part["text"])
    if part_type == "image_url":
        image_url = openai_part.get("image_url")
        if not isinstance(image_url, dict):
            raise ValueError(
                f"Unsupported image content part: {_CONTENT_PART_REPR.repr(openai_part)}. "
                "Image parts must contain an `image_url` object."
            )
        mime_type, base64_image = _parse_openai_base64_data_url(image_url.get("url"))
        return ImageContent(base64_image=base64_image, mime_type=mime_type, detail=image_url.get("detail"))
    if part_type == "file":
        file = openai_part.get("file")
        if not isinstance(file, dict):
            raise ValueError(
                f"Unsupported file content part: {_CONTENT_PART_REPR.repr(openai_part)}. "
                "File parts must contain a `file` object."
            )
        file_data = file.get("file_data")
        if not file_data:
            raise ValueError(
                f"Unsupported file content part: {_CONTENT_PART_REPR.repr(openai_part)}. Only files with inline "
                "base64 `file_data` are supported: files referenced by `file_id` cannot be converted."
            )
        mime_type, base64_data = _parse_openai_base64_data_url(file_data)
        return FileContent(base64_data=base64_data, mime_type=mime_type, filename=file.get("filename"))
    raise ValueError(
        f"Unsupported content part: {_CONTENT_PART_REPR.repr(openai_part)}. "
        "Supported part types are `text`, `image_url`, and `file`."
    )


def _parse_openai_base64_data_url(data_url: Any) -> tuple[str | None, str]:
    """
    Split a base64 data URL in OpenAI format into its MIME type and base64 payload.

    :param data_url: A data URL in the format `data:<mime_type>;base64,<base64_data>`.
    :returns: A tuple containing the MIME type (or None if absent) and the base64 data.
    :raises ValueError: If the URL is not a string or not a base64 data URL.
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError(
            f"Unsupported URL: {_CONTENT_PART_REPR.repr(data_url)}. Only base64 data URLs in the format "
            "`data:<mime_type>;base64,<base64_data>` are supported."
        )
    header, base64_data = data_url.split(";base64,", 1)
    return header[len("data:") :] or None, base64_data


def _haystack_tool_call_to_openai_dict(haystack_tool_call: ToolCall) -> dict[str, Any]:
    """Convert a tool call to OpenAI format, encoding its arguments as JSON."""
    openai_tool_call: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": haystack_tool_call.tool_name,
            "arguments": json.dumps(haystack_tool_call.arguments, ensure_ascii=False),
        },
    }
    if haystack_tool_call.id is not None:
        openai_tool_call["id"] = haystack_tool_call.id
    return openai_tool_call


def _openai_dict_to_haystack_tool_call(openai_tool_call: dict[str, Any]) -> ToolCall:
    """Build a ToolCall from an OpenAI function call, parsing its arguments."""
    if "function" not in openai_tool_call:
        raise ValueError("Tool calls must contain the `function` field")
    return ToolCall(
        id=openai_tool_call.get("id"),
        tool_name=openai_tool_call["function"]["name"],
        arguments=_parse_openai_tool_call_arguments(openai_tool_call["function"].get("arguments")),
    )


def _parse_openai_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    """
    Parse OpenAI-style tool call arguments into a dictionary.

    A dict is returned as is, a missing, null, or empty value becomes `{}`, and anything else is parsed as a
    JSON string.

    :raises json.JSONDecodeError: If the arguments are not valid JSON.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return json.loads(raw_arguments) if raw_arguments else {}


def _parse_openai_text_and_refusal_content(openai_content: str | list[Any] | None, *, role: str) -> dict[str, str]:
    """
    Join the text and refusal content parts of a system, developer, or assistant message in OpenAI format.

    Text parts are joined into a single text because `to_openai_dict_format` only sends the first text of these
    messages. Refusal parts are accepted only in assistant messages and are joined separately, so a refusal never
    ends up in the text.

    :param openai_content: A string, a list of content parts in OpenAI format, or None.
    :param role: The role of the message, which decides whether refusal parts are accepted.
    :returns: A dictionary that maps each part type in `openai_content` to the texts of its parts,
        joined with a newline.
    :raises ValueError: If a content part is not a text part with a `text` string, or, in an assistant message, a
        refusal part with a `refusal` string.
    """
    if isinstance(openai_content, str):
        return {"text": openai_content}
    allowed_types = ("text", "refusal") if role == "assistant" else ("text",)
    texts_by_type: dict[str, list[str]] = {}
    for openai_part in openai_content or []:
        if (
            not isinstance(openai_part, dict)
            or openai_part.get("type") not in allowed_types
            or not isinstance(openai_part.get(openai_part["type"]), str)
        ):
            supported = " and ".join(f"`{t}` parts with a `{t}` string" for t in allowed_types)
            raise ValueError(
                f"Unsupported content part in {role} message: {_CONTENT_PART_REPR.repr(openai_part)}. "
                f"Only {supported} are supported."
            )
        texts_by_type.setdefault(openai_part["type"], []).append(openai_part[openai_part["type"]])
    return {part_type: "\n".join(part_texts) for part_type, part_texts in texts_by_type.items()}
