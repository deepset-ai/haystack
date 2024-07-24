# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .byte_stream import ByteStream


class ChatRole(str, Enum):
    """Enumeration representing the roles within a chat."""

    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"


class ContentType(str, Enum):
    """Enumeration representing the different types of content that fit in a ChatMessage."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"

    @staticmethod
    def valid_byte_stream_types() -> List["ContentType"]:
        """Returns a list of all the valid types of represented by a ByteStream."""
        return [ContentType.IMAGE_BASE64]

    def is_valid_byte_stream_type(self) -> bool:
        """Returns whether the type is a valid type for a ByteStream."""
        return self in ContentType.valid_byte_stream_types()


@dataclass
class ChatMessage:
    """
    Represents a message in a LLM chat conversation.

    :param content: The text content of the message.
    :param role: The role of the entity sending the message.
    :param name: The name of the function being called (only applicable for role FUNCTION).
    :param meta: Additional metadata associated with the message.
    """

    content: Union[str, ByteStream, List[Union[str, ByteStream]]]
    role: ChatRole
    name: Optional[str]
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """
        This method runs after the __init__ method from dataclass.

        It runs some checks on the content of the ChatMessage and populates metadata.
        """
        if isinstance(self.content, str):
            content, content_type = self._parse_string_content(self.content)
            self.content = content
            self.meta["__haystack_content_type__"] = content_type
        elif isinstance(self.content, ByteStream):
            content, content_type = self._parse_byte_stream_content(self.content)
            self.content = content
            self.meta["__haystack_content_type__"] = content_type
        elif isinstance(self.content, list):
            content: List[Union[str, ByteStream]] = []
            content_types = []
            for part in self.content:
                if isinstance(part, str):
                    part_content, content_type = self._parse_string_content(part)
                    content.append(part_content)
                    content_types.append(content_type)
                elif isinstance(part, ByteStream):
                    part_content, content_type = self._parse_byte_stream_content(part)
                    content.append(part_content)
                    content_types.append(content_type)
                else:
                    raise ValueError("Invalid element in content. Valid types are str and ByteStream" "Element: {part}")
            self.content = content
            self.meta["__haystack_content_type__"] = content_types
        else:
            raise ValueError(
                f"Invalid type of content. Valid types are str, "
                f"ByteStream and a list of str and ByteStream objects."
                f"Content: {self.content}"
            )

    @staticmethod
    def _parse_string_content(content: str) -> Tuple[str, ContentType]:
        """
        Parse the string content to differentiate between different types of string representable content.

        :param content: The string content you want to parse.
        :returns: a tuple containing the parsed content and the content type.
        """
        if content.strip().startswith("image_url:"):
            url = content.split("image_url:")[-1].strip()
            return url, ContentType.IMAGE_URL
        else:  # TEXT
            return content, ContentType.TEXT

    @staticmethod
    def _parse_byte_stream_content(content: ByteStream) -> Tuple[ByteStream, ContentType]:
        """
        Parse the byte stream content to differentiate between different types of byte encoded content.

        :param content: The bytes content you want to parse.
        :returns: a tuple containing the parsed content and the content type.

        :raises: Value error if the 'mime_type' attribute is None or any invalid value.
        """
        if content.mime_type is None:
            raise ValueError(
                "Unidentified ByteStream added as part of the content of the ChatMessage."
                "Populate thee 'mime_type' attribute with the identifier of the content type."
            )

        mime_type = content.mime_type.split("/")[0]
        content_type = ContentType(mime_type)
        if not content_type.is_valid_byte_stream_type():
            raise ValueError(
                f"The 'mime_type' attribute of the introduced content "
                f"has a not valid ContentType for a ByteStream"
                f"Value: {content_type}. Valid content types:"
                + ", ".join([c.value for c in ContentType.valid_byte_stream_types()])
            )

        return content, content_type

    def get_content_types(self) -> Union[ContentType, List[ContentType]]:
        """Returns the content of the '__haystack_content_type__' meta key."""
        return self.meta["__haystack_content_type__"]

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

        types = self.get_content_types()
        if isinstance(types, list) and isinstance(self.content, list):
            content: List[Dict[str, Any]] = []
            for type_, part in zip(types, self.content):
                if type_ is ContentType.TEXT and isinstance(part, str):
                    content.append({"type": "text", "text": part})
                elif type_ is ContentType.IMAGE_URL and isinstance(part, str):
                    content.append({"type": "image_url", "image_url": {"url": part}})
                elif type_ is ContentType.IMAGE_BASE64 and isinstance(part, ByteStream):
                    file_encoding = part.mime_type.split("/")[-1]

                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{file_encoding};base64,{part.to_string()}"},
                        }
                    )
                else:
                    raise ValueError("The content types stored at metadata '__haystack_content_type__' was corrupted.")
        else:
            if types is ContentType.TEXT and isinstance(self.content, str):
                content: str = self.content
            elif types is ContentType.IMAGE_URL and isinstance(self.content, str):
                content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": self.content}}]
            elif types is ContentType.IMAGE_BASE64 and isinstance(self.content, ByteStream):
                file_encoding = self.content.mime_type.split("/")[-1]

                content: List[Dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{file_encoding};base64,{self.content.to_string()}"},
                    }
                ]
            else:
                raise ValueError("The content types stored at metadata '__haystack_content_type__' was corrupted.")

        msg["content"] = content
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
        cls, content: Union[str, ByteStream, List[Union[str, ByteStream]]], meta: Optional[Dict[str, Any]] = None
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param content: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.ASSISTANT, None, meta or {})

    @classmethod
    def from_user(cls, content: Union[str, ByteStream, List[Union[str, ByteStream]]]) -> "ChatMessage":
        """
        Create a message from the user.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.USER, None)

    @classmethod
    def from_system(cls, content: Union[str, ByteStream, List[Union[str, ByteStream]]]) -> "ChatMessage":
        """
        Create a message from the system.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content, ChatRole.SYSTEM, None)

    @classmethod
    def from_function(cls, content: Union[str, ByteStream, List[Union[str, ByteStream]]], name: str) -> "ChatMessage":
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
        if "__haystack_content_type__" in data["meta"]:
            types = data["meta"].pop("__haystack_content_type__")
            if isinstance(types, list) and isinstance(self.content, list):
                content = []
                for type_, part in zip(types, self.content):
                    if type_ is ContentType.IMAGE_URL and isinstance(part, str):
                        full_part: Union[str, Dict[str, Any]] = f"image_url:{part}"
                    elif type_ in ContentType.valid_byte_stream_types() and isinstance(part, ByteStream):
                        full_part: Union[str, Dict[str, Any]] = asdict(part)
                    else:
                        full_part: Union[str, Dict[str, Any]] = part
                    content.append(full_part)

                data["content"] = content
            else:
                if types is ContentType.IMAGE_URL and isinstance(data["content"], str):
                    data["content"] = f"image_url:{data['content']}"
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
            if isinstance(data["content"], str):
                pass
            elif isinstance(data["content"], dict):
                data["content"] = ByteStream(**data["content"])
            elif isinstance(data["content"], list):
                content: List[Union[str, ByteStream]] = []
                for part in data["content"]:
                    if isinstance(part, str):
                        content.append(part)
                    elif isinstance(part, dict):
                        content.append(ByteStream(**part))
                    else:
                        raise ValueError("Invalid dict contains non deserializable content.")
                data["content"] = content
            else:
                raise ValueError("Invalid dict contains non deserializable content.")

        return cls(**data)
