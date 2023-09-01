from dataclasses import dataclass


@dataclass
class ChatMessage:
    content: str
    role: str
