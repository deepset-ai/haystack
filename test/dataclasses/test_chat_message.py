# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer

from haystack.dataclasses import ChatMessage, ChatRole, ContentPart


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


def test_to_openai_format_with_text_content():
    message = ChatMessage.from_system("You are good assistant")
    assert message.to_openai_format() == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert message.to_openai_format() == {"role": "user", "content": "I have a question"}

    message = ChatMessage.from_function("Function call", "function_name")
    assert message.to_openai_format() == {"role": "function", "content": "Function call", "name": "function_name"}


def test_to_openai_format_with_content_part():
    message = ChatMessage.from_system(ContentPart.from_text("Content"))
    assert message.to_openai_format() == {"role": "system", "content": {"type": "text", "text": "Content"}}

    message = ChatMessage.from_user(ContentPart.from_image_url("image.com/test.jpg"))
    assert message.to_openai_format() == {
        "role": "user",
        "content": {"type": "image_url", "image_url": {"url": "image.com/test.jpg"}},
    }


def test_to_openai_format_with_list_content():
    message = ChatMessage.from_assistant(
        content=["String Content", ContentPart.from_image_url("images.com/test.jpg"), ContentPart.from_text("Content")]
    )

    assert message.to_openai_format() == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "String Content"},
            {"type": "image_url", "image_url": {"url": "images.com/test.jpg"}},
            {"type": "text", "text": "Content"},
        ],
    }


@pytest.mark.integration
def test_apply_chat_templating_on_chat_message():
    messages = [ChatMessage.from_system("You are good assistant"), ChatMessage.from_user("I have a question")]
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    formatted_messages = [m.to_openai_format() for m in messages]
    tokenized_messages = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
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
    formatted_messages = [m.to_openai_format() for m in messages]
    tokenized_messages = tokenizer.apply_chat_template(
        formatted_messages, chat_template=anthropic_template, tokenize=False
    )
    assert tokenized_messages == "You are good assistant\nHuman: I have a question\nAssistant:"


def test_to_dict_with_text_content():
    message = ChatMessage.from_user("content")
    message.meta["some"] = "some"

    assert message.to_dict() == {"content": "content", "role": "user", "name": None, "meta": {"some": "some"}}


def test_to_dict_with_content_part():
    message = ChatMessage.from_user(ContentPart.from_text("content"))

    assert message.to_dict() == {
        "content": {"type": "text", "content": "content"},
        "role": "user",
        "name": None,
        "meta": {},
    }


def test_to_dict_with_list_content():
    message = ChatMessage.from_user(
        content=[ContentPart.from_text("Content"), ContentPart.from_image_url("image.com/test.jpg"), "String Content"]
    )

    assert message.to_dict() == {
        "content": [
            {"type": "text", "content": "Content"},
            {"type": "image_url", "content": "image.com/test.jpg"},
            "String Content",
        ],
        "role": "user",
        "name": None,
        "meta": {},
    }


def test_from_dict_with_text_content():
    assert ChatMessage.from_dict(data={"content": "text", "role": "user", "name": None}) == ChatMessage(
        content="text", role=ChatRole("user"), name=None, meta={}
    )


def test_from_dict_with_content_part():
    assert ChatMessage.from_dict(
        data={"content": {"type": "text", "content": "content"}, "role": "user", "name": None, "meta": {}}
    ) == ChatMessage.from_user(content=ContentPart.from_text("content"))


def test_from_dict_with_meta():
    assert ChatMessage.from_dict(
        data={"content": "text", "role": "user", "name": None, "meta": {"something": "something"}}
    ) == ChatMessage(content="text", role=ChatRole("user"), name=None, meta={"something": "something"})


def test_from_dict_with_list_content():
    assert ChatMessage.from_dict(
        {
            "content": [
                {"type": "text", "content": "Content"},
                {"type": "image_url", "content": "image.com/test.jpg"},
                "String Content",
            ],
            "role": "user",
            "name": None,
            "meta": {},
        }
    ) == ChatMessage.from_user(
        content=[ContentPart.from_text("Content"), ContentPart.from_image_url("image.com/test.jpg"), "String Content"]
    )
