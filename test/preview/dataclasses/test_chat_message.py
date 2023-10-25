import pytest

from haystack.preview.dataclasses import ChatMessage, ChatRole


@pytest.mark.unit
def test_from_assistant_with_valid_content():
    content = "Hello, how can I assist you?"
    message = ChatMessage.from_assistant(content)
    assert message.content == content
    assert message.role == ChatRole.ASSISTANT


@pytest.mark.unit
def test_from_user_with_valid_content():
    content = "I have a question."
    message = ChatMessage.from_user(content)
    assert message.content == content
    assert message.role == ChatRole.USER


@pytest.mark.unit
def test_from_system_with_valid_content():
    content = "System message."
    message = ChatMessage.from_system(content)
    assert message.content == content
    assert message.role == ChatRole.SYSTEM


@pytest.mark.unit
def test_with_empty_content():
    message = ChatMessage("", ChatRole.USER, None)
    assert message.content == ""


@pytest.mark.unit
def test_with_invalid_role():
    with pytest.raises(TypeError):
        ChatMessage("Invalid role", "invalid_role")


@pytest.mark.unit
def test_from_function_with_empty_name():
    content = "Function call"
    message = ChatMessage.from_function(content, "")
    assert message.content == content
    assert message.name == ""
