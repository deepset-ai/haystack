# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .byte_stream import ByteStream
from .content_part import ContentPart


class ChatRole(str, Enum):
    """Enumeration representing the roles within a chat."""

    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """
    Represents a message in a LLM chat conversation.

    :param content: The content of the message.
    :param role: The role of the entity sending the message.
    :param name: The name of the function being called (only applicable for role FUNCTION).
    :param meta: Additional metadata associated with the message.
    """

    content: Union[str, ContentPart, List[Union[str, ContentPart]]]
    role: ChatRole
    name: Optional[str]
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the message to the format expected by OpenAI's Chat API.

        See the [API reference](https://platform.openai.com/docs/api-reference/chat/create) for details.

        :returns: A dictionary with the following key:
            - `role`
            - `content`
            - `name` (optional)
        """
        msg = {"role": self.role.value}
        if self.name:
            msg["name"] = self.name

        if isinstance(self.content, str):
            msg["content"] = self.content
        elif isinstance(self.content, ContentPart):
            msg["content"] = self.content.to_openai_format()
        elif isinstance(self.content, list):
            msg["content"] = []
            for part in self.content:
                if isinstance(part, str):
                    msg["content"].append(ContentPart.from_text(part).to_openai_format())
                elif isinstance(part, ContentPart):
                    msg["content"].append(part.to_openai_format())
                else:
                    raise ValueError(
                        "One of the elements of the content is not of a valid type."
                        "Valid types: str or ContentPart. Element: {part}"
                    )
        else:
            raise ValueError(
                "The content of the message is not of a valid type."
                "Valid types: str, ContentPart or list of str and ContentPart."
                "Content: {self.content}"
            )

        return msg

    def is_from(self, role: ChatRole) -> bool:
        """
        Check if the message is from a specific role.

        :param role: The role to check against.
        :returns: True if the message is from the specified role, False otherwise.
        """
        return self.role == role

    @classmethod
    def from_assistant(
        cls, content: Union[str, ContentPart, List[Union[str, ContentPart]]], meta: Optional[Dict[str, Any]] = None
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param content: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.ASSISTANT, None, meta or {})

    @classmethod
    def from_user(cls, content: Union[str, ContentPart, List[Union[str, ContentPart]]]) -> "ChatMessage":
        """
        Create a message from the user.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.USER, None)

    @classmethod
    def from_system(cls, content: Union[str, ContentPart, List[Union[str, ContentPart]]]) -> "ChatMessage":
        """
        Create a message from the system.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.SYSTEM, None)

    @classmethod
    def from_function(cls, content: Union[str, ContentPart, List[Union[str, ContentPart]]], name: str) -> "ChatMessage":
        """
        Create a message from a function call.

        :param content: The text content of the message.
        :param name: The name of the function being called.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.FUNCTION, name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """
        data = asdict(self)
        data["role"] = self.role.value

        if isinstance(self.content, str):
            data["content"] = self.content
        elif isinstance(self.content, ContentPart):
            data["content"] = self.content.to_dict()
        elif isinstance(self.content, list):
            data["content"] = []
            for part in self.content:
                if isinstance(part, str):
                    data["content"].append(part)
                elif isinstance(part, ContentPart):
                    data["content"].append(part.to_dict())
                else:
                    raise ValueError(
                        "One of the elements of the content is not of a valid type."
                        "Valid types: str or ContentPart. Element: {part}"
                    )
        else:
            raise ValueError(
                "The content of the message is not of a valid type."
                "Valid types: str, ContentPart or list of str and ContentPart."
                "Content: {self.content}"
            )

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """
        Creates a new ChatMessage object from a dictionary.

        :param data:
            The dictionary to build the ChatMessage object.
        :returns:
            The created object.
        """
        data["role"] = ChatRole(data["role"])

        if "content" in data:
            if isinstance(data["content"], dict):  # Assume it is a ContentPart
                data["content"] = ContentPart.from_dict(data["content"])
            elif isinstance(data["content"], list):
                content = data.pop("content")
                data["content"] = []
                for part in content:
                    if isinstance(part, str):
                        data["content"].append(part)
                    else:
                        data["content"].append(ContentPart.from_dict(part))

        return cls(**data)
