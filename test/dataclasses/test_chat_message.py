import pytest
from transformers import AutoTokenizer

from haystack.dataclasses import ChatMessage, ChatRole


def test_from_assistant_with_valid_content():
    content = "Hello, how can I assist you?"
    message = ChatMessage.from_assistant(content)
    assert message.content == content
    assert message.role == ChatRole.ASSISTANT


def test_from_user_with_valid_content():
    content = "I have a question."
    message = ChatMessage.from_user(content)
    assert message.content == content
    assert message.role == ChatRole.USER


def test_from_system_with_valid_content():
    content = "System message."
    message = ChatMessage.from_system(content)
    assert message.content == content
    assert message.role == ChatRole.SYSTEM


def test_with_empty_content():
    message = ChatMessage.from_user("")
    assert message.content == ""


def test_from_function_with_empty_name():
    content = "Function call"
    message = ChatMessage.from_function(content, "")
    assert message.content == content
    assert message.name == ""


@pytest.mark.integration
def test_apply_chat_templating_on_chat_message():
    messages = [ChatMessage.from_system("You are good assistant"), ChatMessage.from_user("I have a question")]
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenized_messages = tokenizer.apply_chat_template(messages, tokenize=False)
    assert tokenized_messages == "<|system|>\nYou are good assistant</s>\n<|user|>\nI have a question</s>\n"


@pytest.mark.integration
def test_apply_custom_chat_templating_on_chat_message():
    anthropic_template = (
        "{%- for message in messages %}"
        "{%- if message.role == 'user' %}\n\nHuman: {{ message.content.strip() }}"
        "{%- elif message.role == 'assistant' %}\n\nAssistant: {{ message.content.strip() }}"
        "{%- elif message.role == 'function' %}{{ raise('anthropic does not support function calls.') }}"
        "{%- elif message.role == 'system' and loop.index == 1 %}{{ message.content }}"
        "{%- else %}{{ raise('Invalid message role: ' + message.role) }}"
        "{%- endif %}"
        "{%- endfor %}"
        "\n\nAssistant:"
    )
    messages = [ChatMessage.from_system("You are good assistant"), ChatMessage.from_user("I have a question")]
    # could be any tokenizer, let's use the one we already likely have in cache
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenized_messages = tokenizer.apply_chat_template(messages, chat_template=anthropic_template, tokenize=False)
    assert tokenized_messages == "You are good assistant\nHuman: I have a question\nAssistant:"
