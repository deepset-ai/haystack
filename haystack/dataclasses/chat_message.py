# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from haystack import logging
from haystack.dataclasses.image_content import ImageContent

logger = logging.getLogger(__name__)


LEGACY_INIT_PARAMETERS = {"role", "content", "meta", "name"}


class ChatRole(str, Enum):
    """
    Enumeration representing the roles within a chat.
    """

    #: The user role. A message from the user contains only text.
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


@dataclass
class ToolCall:
    """
    Represents a Tool call prepared by the model, usually contained in an assistant message.

    :param id: The ID of the Tool call.
    :param tool_name: The name of the Tool to call.
    :param arguments: The arguments to call the Tool with.
    """

    tool_name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None  # noqa: A003

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ToolCall into a dictionary.

        :returns: A dictionary with keys 'tool_name', 'arguments', and 'id'.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """
        Creates a new ToolCall object from a dictionary.

        :param data:
            The dictionary to build the ToolCall object.
        :returns:
            The created object.
        """
        return ToolCall(**data)


@dataclass
class ToolCallResult:
    """
    Represents the result of a Tool invocation.

    :param result: The result of the Tool invocation.
    :param origin: The Tool call that produced this result.
    :param error: Whether the Tool invocation resulted in an error.
    """

    result: str
    origin: ToolCall
    error: bool

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ToolCallResult into a dictionary.

        :returns: A dictionary with keys 'result', 'origin', and 'error'.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallResult":
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
        return ToolCallResult(result=data["result"], origin=ToolCall.from_dict(data["origin"]), error=data["error"])


@dataclass
class TextContent:
    """
    The textual content of a chat message.

    :param text: The text content of the message.
    """

    text: str


ChatMessageContentT = Union[TextContent, ToolCall, ToolCallResult, ImageContent]


def _deserialize_content_part(part: Dict[str, Any]) -> ChatMessageContentT:
    """
    Deserialize a single content part of a serialized ChatMessage.

    :param part:
        A dictionary representing a single content part of a serialized ChatMessage.
    :returns:
        A ChatMessageContentT object.
    :raises ValueError:
        If the part is not a valid ChatMessageContentT object.
    """
    if "text" in part:
        return TextContent(text=part["text"])
    if "tool_call" in part:
        return ToolCall(**part["tool_call"])
    if "tool_call_result" in part:
        result = part["tool_call_result"]["result"]
        origin = ToolCall(**part["tool_call_result"]["origin"])
        error = part["tool_call_result"]["error"]
        tcr = ToolCallResult(result=result, origin=origin, error=error)
        return tcr
    if "image" in part:
        return ImageContent(**part["image"])
    raise ValueError(f"Unsupported content part in the serialized ChatMessage: `{part}`")


def _deserialize_content(serialized_content: List[Dict[str, Any]]) -> List[ChatMessageContentT]:
    """
    Deserialize the `content` field of a serialized ChatMessage.

    :param serialized_content:
        The `content` field of a serialized ChatMessage (a list of dictionaries).

    :returns:
        Deserialized `content` field as a list of `ChatMessageContentT` objects.
    """
    return [_deserialize_content_part(part) for part in serialized_content]


def _serialize_content_part(part: ChatMessageContentT) -> Dict[str, Any]:
    """
    Serialize a single content part of a ChatMessage.

    :param part:
        A ChatMessageContentT object.
    :returns:
        A dictionary representing the content part.
    :raises TypeError:
        If the part is not a valid ChatMessageContentT object.
    """
    if isinstance(part, TextContent):
        return {"text": part.text}
    elif isinstance(part, ToolCall):
        return {"tool_call": asdict(part)}
    elif isinstance(part, ToolCallResult):
        return {"tool_call_result": asdict(part)}
    elif isinstance(part, ImageContent):
        return {"image": asdict(part)}
    raise TypeError(f"Unsupported type in ChatMessage content: `{type(part).__name__}` for `{part}`.")


@dataclass
class ChatMessage:
    """
    Represents a message in a LLM chat conversation.

    Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a ChatMessage.
    """

    _role: ChatRole
    _content: Sequence[ChatMessageContentT]
    _name: Optional[str] = None
    _meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __new__(cls, *args, **kwargs):
        """
        This method is reimplemented to make the changes to the `ChatMessage` dataclass more visible.
        """

        general_msg = (
            "Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a "
            "ChatMessage. For more information about the new API and how to migrate, see the documentation:"
            " https://docs.haystack.deepset.ai/docs/chatmessage"
        )

        if any(param in kwargs for param in LEGACY_INIT_PARAMETERS):
            raise TypeError(
                "The `role`, `content`, `meta`, and `name` init parameters of `ChatMessage` have been removed. "
                f"{general_msg}"
            )

        allowed_content_types = (TextContent, ToolCall, ToolCallResult)
        if len(args) > 1 and not isinstance(args[1], allowed_content_types):
            raise TypeError(
                "The `_content` parameter of `ChatMessage` must be one of the following types: "
                f"{', '.join(t.__name__ for t in allowed_content_types)}. "
                f"{general_msg}"
            )

        return super(ChatMessage, cls).__new__(cls)

    def __getattribute__(self, name):
        """
        This method is reimplemented to make the `content` attribute removal more visible.
        """

        if name == "content":
            msg = (
                "The `content` attribute of `ChatMessage` has been removed. "
                "Use the `text` property to access the textual value. "
                "For more information about the new API and how to migrate, see the documentation: "
                "https://docs.haystack.deepset.ai/docs/chatmessage"
            )
            raise AttributeError(msg)
        return object.__getattribute__(self, name)

    def __len__(self):
        return len(self._content)

    @property
    def role(self) -> ChatRole:
        """
        Returns the role of the entity sending the message.
        """
        return self._role

    @property
    def meta(self) -> Dict[str, Any]:
        """
        Returns the metadata associated with the message.
        """
        return self._meta

    @property
    def name(self) -> Optional[str]:
        """
        Returns the name associated with the message.
        """
        return self._name

    @property
    def texts(self) -> List[str]:
        """
        Returns the list of all texts contained in the message.
        """
        return [content.text for content in self._content if isinstance(content, TextContent)]

    @property
    def text(self) -> Optional[str]:
        """
        Returns the first text contained in the message.
        """
        if texts := self.texts:
            return texts[0]
        return None

    @property
    def tool_calls(self) -> List[ToolCall]:
        """
        Returns the list of all Tool calls contained in the message.
        """
        return [content for content in self._content if isinstance(content, ToolCall)]

    @property
    def tool_call(self) -> Optional[ToolCall]:
        """
        Returns the first Tool call contained in the message.
        """
        if tool_calls := self.tool_calls:
            return tool_calls[0]
        return None

    @property
    def tool_call_results(self) -> List[ToolCallResult]:
        """
        Returns the list of all Tool call results contained in the message.
        """
        return [content for content in self._content if isinstance(content, ToolCallResult)]

    @property
    def tool_call_result(self) -> Optional[ToolCallResult]:
        """
        Returns the first Tool call result contained in the message.
        """
        if tool_call_results := self.tool_call_results:
            return tool_call_results[0]
        return None

    @property
    def images(self) -> List[ImageContent]:
        """
        Returns the list of all images contained in the message.
        """
        return [content for content in self._content if isinstance(content, ImageContent)]

    @property
    def image(self) -> Optional[ImageContent]:
        """
        Returns the first image contained in the message.
        """
        if images := self.images:
            return images[0]
        return None

    def is_from(self, role: Union[ChatRole, str]) -> bool:
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
        text: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        *,
        content_parts: Optional[Sequence[Union[TextContent, str, ImageContent]]] = None,
    ) -> "ChatMessage":
        """
        Create a message from the user.

        :param text: The text content of the message. Specify this or content_parts.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :param content_parts: A list of content parts to include in the message. Specify this or text.
        :returns: A new ChatMessage instance.
        """
        if text is None and content_parts is None:
            raise ValueError("Either text or content_parts must be provided.")
        if text is not None and content_parts is not None:
            raise ValueError("Only one of text or content_parts can be provided.")

        content: Sequence[Union[TextContent, ImageContent]] = []

        if text is not None:
            content = [TextContent(text=text)]
        elif content_parts is not None:
            content = [TextContent(el) if isinstance(el, str) else el for el in content_parts]
            if not any(isinstance(el, TextContent) for el in content):
                raise ValueError("The user message must contain at least one textual part.")

            unsupported_parts = [el for el in content if not isinstance(el, (ImageContent, TextContent))]
            if unsupported_parts:
                raise ValueError(
                    f"The user message must contain only text or image parts. Unsupported parts: {unsupported_parts}"
                )

        return cls(_role=ChatRole.USER, _content=content, _meta=meta or {}, _name=name)

    @classmethod
    def from_system(cls, text: str, meta: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> "ChatMessage":
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
        text: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param tool_calls: The Tool calls to include in the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :returns: A new ChatMessage instance.
        """
        content: List[ChatMessageContentT] = []
        if text is not None:
            content.append(TextContent(text=text))
        if tool_calls:
            content.extend(tool_calls)

        return cls(_role=ChatRole.ASSISTANT, _content=content, _meta=meta or {}, _name=name)

    @classmethod
    def from_tool(
        cls, tool_result: str, origin: ToolCall, error: bool = False, meta: Optional[Dict[str, Any]] = None
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """

        serialized: Dict[str, Any] = {}
        serialized["role"] = self._role.value
        serialized["meta"] = self._meta
        serialized["name"] = self._name

        serialized["content"] = [_serialize_content_part(part) for part in self._content]
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """
        Creates a new ChatMessage object from a dictionary.

        :param data:
            The dictionary to build the ChatMessage object.
        :returns:
            The created object.
        """
        if not "role" in data and not "_role" in data:
            raise ValueError(
                "The `role` field is required in the message dictionary. "
                f"Expected a dictionary with 'role' field containing one of: {[role.value for role in ChatRole]}. "
                f"Common roles are 'user' (for user messages) and 'assistant' (for AI responses). "
                f"Received dictionary with keys: {list(data.keys())}"
            )

        if "content" in data:
            init_params: Dict[str, Any] = {
                "_role": ChatRole(data["role"]),
                "_name": data.get("name"),
                "_meta": data.get("meta") or {},
            }

            if isinstance(data["content"], list):
                # current format - the serialized `content` field is a list of dictionaries
                init_params["_content"] = _deserialize_content(data["content"])
            elif isinstance(data["content"], str):
                # pre 2.9.0 format - the `content` field is a string
                init_params["_content"] = [TextContent(text=data["content"])]
            else:
                raise TypeError(f"Unsupported content type in serialized ChatMessage: `{(data['content'])}`")
            return cls(**init_params)

        if "_content" in data:
            # format for versions >=2.9.0 and <2.12.0 - the serialized `_content` field is a list of dictionaries
            return cls(
                _role=ChatRole(data["_role"]),
                _content=_deserialize_content(data["_content"]),
                _name=data.get("_name"),
                _meta=data.get("_meta") or {},
            )

        raise ValueError(f"Missing 'content' or '_content' in serialized ChatMessage: `{data}`")

    def to_openai_dict_format(self, require_tool_call_ids: bool = True) -> Dict[str, Any]:
        """
        Convert a ChatMessage to the dictionary format expected by OpenAI's Chat API.

        :param require_tool_call_ids:
            If True (default), enforces that each Tool Call includes a non-null `id` attribute.
            Set to False to allow Tool Calls without `id`, which may be suitable for shallow OpenAI-compatible APIs.
        :returns:
            The ChatMessage in the format expected by OpenAI's Chat API.

        :raises ValueError:
            If the message format is invalid, or if `require_tool_call_ids` is True and any Tool Call is missing an
            `id` attribute.
        """
        text_contents = self.texts
        tool_calls = self.tool_calls
        tool_call_results = self.tool_call_results

        if not text_contents and not tool_calls and not tool_call_results:
            raise ValueError(
                "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
            )
        if len(text_contents) + len(tool_call_results) > 1:
            raise ValueError(
                "For OpenAI compatibility, a `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`."
            )

        openai_msg: Dict[str, Any] = {"role": self._role.value}

        # Add name field if present
        if self._name is not None:
            openai_msg["name"] = self._name

        # user message
        if openai_msg["role"] == "user":
            if len(self._content) == 1:
                openai_msg["content"] = self.text
                return openai_msg

            # if the user message contains a list of text and images, OpenAI expects a list of dictionaries
            content = []
            for part in self._content:
                if isinstance(part, TextContent):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    image_item: Dict[str, Any] = {
                        "type": "image_url",
                        # If no MIME type is provided, default to JPEG.
                        # OpenAI API appears to tolerate MIME type mismatches.
                        "image_url": {"url": f"data:{part.mime_type or 'image/jpeg'};base64,{part.base64_image}"},
                    }
                    if part.detail:
                        image_item["image_url"]["detail"] = part.detail
                    content.append(image_item)
            openai_msg["content"] = content
            return openai_msg

        # tool message
        if tool_call_results:
            result = tool_call_results[0]
            openai_msg["content"] = result.result
            if result.origin.id is not None:
                openai_msg["tool_call_id"] = result.origin.id
            elif require_tool_call_ids:
                raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
            # OpenAI does not provide a way to communicate errors in tool invocations, so we ignore the error field
            return openai_msg

        # system and assistant messages
        if text_contents:
            openai_msg["content"] = text_contents[0]
        if tool_calls:
            openai_tool_calls = []
            for tc in tool_calls:
                openai_tool_call = {
                    "type": "function",
                    # We disable ensure_ascii so special chars like emojis are not converted
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                if tc.id is not None:
                    openai_tool_call["id"] = tc.id
                elif require_tool_call_ids:
                    raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
                openai_tool_calls.append(openai_tool_call)
            openai_msg["tool_calls"] = openai_tool_calls
        return openai_msg

    @staticmethod
    def _validate_openai_message(message: Dict[str, Any]) -> None:
        """
        Validate that a message dictionary follows OpenAI's Chat API format.

        :param message: The message dictionary to validate
        :raises ValueError: If the message format is invalid
        """
        if "role" not in message:
            raise ValueError("The `role` field is required in the message dictionary.")

        role = message["role"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if role not in ["assistant", "user", "system", "developer", "tool"]:
            raise ValueError(f"Unsupported role: {role}")

        if role == "assistant":
            if not content and not tool_calls:
                raise ValueError("For assistant messages, either `content` or `tool_calls` must be present.")
            if tool_calls:
                for tc in tool_calls:
                    if "function" not in tc:
                        raise ValueError("Tool calls must contain the `function` field")
        elif not content:
            raise ValueError(f"The `content` field is required for {role} messages.")

    @classmethod
    def from_openai_dict_format(cls, message: Dict[str, Any]) -> "ChatMessage":
        """
        Create a ChatMessage from a dictionary in the format expected by OpenAI's Chat API.

        NOTE: While OpenAI's API requires `tool_call_id` in both tool calls and tool messages, this method
        accepts messages without it to support shallow OpenAI-compatible APIs.
        If you plan to use the resulting ChatMessage with OpenAI, you must include `tool_call_id` or you'll
        encounter validation errors.

        :param message:
            The OpenAI dictionary to build the ChatMessage object.
        :returns:
            The created ChatMessage object.

        :raises ValueError:
            If the message dictionary is missing required fields.
        """
        cls._validate_openai_message(message)

        role = message["role"]
        content = message.get("content")
        name = message.get("name")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")

        if role == "assistant":
            haystack_tool_calls = None
            if tool_calls:
                haystack_tool_calls = []
                for tc in tool_calls:
                    haystack_tc = ToolCall(
                        id=tc.get("id"),
                        tool_name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    haystack_tool_calls.append(haystack_tc)
            return cls.from_assistant(text=content, name=name, tool_calls=haystack_tool_calls)

        assert content is not None  # ensured by _validate_openai_message, but we need to make mypy happy

        if role == "user":
            return cls.from_user(text=content, name=name)
        if role in ["system", "developer"]:
            return cls.from_system(text=content, name=name)

        return cls.from_tool(
            tool_result=content, origin=ToolCall(id=tool_call_id, tool_name="", arguments={}), error=False
        )
