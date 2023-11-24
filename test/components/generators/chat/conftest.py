import pytest

from haystack.dataclasses import ChatMessage


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]
