from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


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

    :param content: The text content of the message.
    :param role: The role of the entity sending the message.
    :param name: The name of the function being called (only applicable for role FUNCTION).
    :param metadata: Additional metadata associated with the message.
    """

    content: str
    role: ChatRole
    name: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    def is_from(self, role: ChatRole) -> bool:
        """
        Check if the message is from a specific role.

        :param role: The role to check against.
        :return: True if the message is from the specified role, False otherwise.
        """
        return self.role == role

    @classmethod
    def from_assistant(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param content: The text content of the message.
        :param metadata: Additional metadata associated with the message.
        :return: A new ChatMessage instance.
        """
        return cls(content, ChatRole.ASSISTANT, None, metadata or {})

    @classmethod
    def from_user(cls, content: str) -> "ChatMessage":
        """
        Create a message from the user.

        :param content: The text content of the message.
        :return: A new ChatMessage instance.
        """
        return cls(content, ChatRole.USER, None)

    @classmethod
    def from_system(cls, content: str) -> "ChatMessage":
        """
        Create a message from the system.

        :param content: The text content of the message.
        :return: A new ChatMessage instance.
        """
        return cls(content, ChatRole.SYSTEM, None)

    @classmethod
    def from_function(cls, content: str, name: str) -> "ChatMessage":
        """
        Create a message from a function call.

        :param content: The text content of the message.
        :param name: The name of the function being called.
        :return: A new ChatMessage instance.
        """
        return cls(content, ChatRole.FUNCTION, name)
