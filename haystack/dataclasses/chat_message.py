# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

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

    #: The function role. Deprecated in favor of `TOOL`.
    FUNCTION = "function"

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


@dataclass
class TextContent:
    """
    The textual content of a chat message.

    :param text: The text content of the message.
    """

    text: str


ChatMessageContentT = Union[TextContent, ToolCall, ToolCallResult]


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
            " https://docs.haystack.deepset.ai/docs/data-classes#chatmessage"
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

    def __post_init__(self):
        if self._role == ChatRole.FUNCTION:
            msg = "The `FUNCTION` role has been deprecated in favor of `TOOL` and will be removed in 2.10.0. "
            warnings.warn(msg, DeprecationWarning)

    def __getattribute__(self, name):
        """
        This method is reimplemented to make the `content` attribute removal more visible.
        """

        if name == "content":
            msg = (
                "The `content` attribute of `ChatMessage` has been removed. "
                "Use the `text` property to access the textual value. "
                "For more information about the new API and how to migrate, see the documentation: "
                "https://docs.haystack.deepset.ai/docs/data-classes#chatmessage"
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
    def from_user(cls, text: str, meta: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> "ChatMessage":
        """
        Create a message from the user.

        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :returns: A new ChatMessage instance.
        """
        return cls(_role=ChatRole.USER, _content=[TextContent(text=text)], _meta=meta or {}, _name=name)

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

    @classmethod
    def from_function(cls, content: str, name: str) -> "ChatMessage":
        """
        Create a message from a function call. Deprecated in favor of `from_tool`.

        :param content: The text content of the message.
        :param name: The name of the function being called.
        :returns: A new ChatMessage instance.
        """
        msg = (
            "The `from_function` method is deprecated and will be removed in version 2.10.0. "
            "Its behavior has changed: it now attempts to convert legacy function messages to tool messages. "
            "This conversion is not guaranteed to succeed in all scenarios. "
            "Please migrate to `ChatMessage.from_tool` and carefully verify the results if you "
            "continue to use this method."
        )
        warnings.warn(msg)

        return cls.from_tool(content, ToolCall(id=None, tool_name=name, arguments={}), error=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """
        serialized: Dict[str, Any] = {}
        serialized["_role"] = self._role.value
        serialized["_meta"] = self._meta
        serialized["_name"] = self._name
        content: List[Dict[str, Any]] = []
        for part in self._content:
            if isinstance(part, TextContent):
                content.append({"text": part.text})
            elif isinstance(part, ToolCall):
                content.append({"tool_call": asdict(part)})
            elif isinstance(part, ToolCallResult):
                content.append({"tool_call_result": asdict(part)})
            else:
                raise TypeError(f"Unsupported type in ChatMessage content: `{type(part).__name__}` for `{part}`.")

        serialized["_content"] = content
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
        if any(param in data for param in LEGACY_INIT_PARAMETERS):
            raise TypeError(
                "The `role`, `content`, `meta`, and `name` init parameters of `ChatMessage` have been removed. "
                "For more information about the new API and how to migrate, see the documentation: "
                "https://docs.haystack.deepset.ai/docs/data-classes#chatmessage"
            )

        data["_role"] = ChatRole(data["_role"])

        content: List[ChatMessageContentT] = []

        for part in data["_content"]:
            if "text" in part:
                content.append(TextContent(text=part["text"]))
            elif "tool_call" in part:
                content.append(ToolCall(**part["tool_call"]))
            elif "tool_call_result" in part:
                result = part["tool_call_result"]["result"]
                origin = ToolCall(**part["tool_call_result"]["origin"])
                error = part["tool_call_result"]["error"]
                tcr = ToolCallResult(result=result, origin=origin, error=error)
                content.append(tcr)
            else:
                raise ValueError(f"Unsupported content in serialized ChatMessage: `{part}`")

        data["_content"] = content

        return cls(**data)
