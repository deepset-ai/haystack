from typing import Any, Dict, List, Optional, Protocol

from haystack.dataclasses import ChatMessage


class ChatGenerator(Protocol):
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Generate responses for a list of messages.

        :param messages: List of messages to generate responses for.
        :param generation_kwargs: Optional additional arguments for the generation process.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ChatGenerator instance into a dictionary representation.
        """
        ...

    def from_dict(cls, config: Dict[str, Any]) -> "ChatGenerator":
        """
        Create a ChatGenerator instance from a dictionary representation.

        :param config: Dictionary containing the configuration of the ChatGenerator.
        """
        ...
