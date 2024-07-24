# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer

from haystack.dataclasses import ChatMessage, ChatRole, ContentType, ByteStream


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


def test_to_openai_format():
    message = ChatMessage.from_system("You are good assistant")
    assert message.to_openai_format() == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert message.to_openai_format() == {"role": "user", "content": "I have a question"}

    message = ChatMessage.from_function("Function call", "function_name")
    assert message.to_openai_format() == {"role": "function", "content": "Function call", "name": "function_name"}


def test_to_openai_format_with_multimodal_content():
    message = ChatMessage.from_system("image_url:images.com/test.jpg")
    assert message.to_openai_format() == {
        "role": "system",
        "content": [{"type": "image_url", "image_url": {"url": "images.com/test.jpg"}}],
    }

    message = ChatMessage.from_user(ByteStream.from_base64_image(b"image"))
    assert message.to_openai_format() == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/jpg;base64,image"}}],
    }

    message = ChatMessage.from_user(ByteStream.from_base64_image(b"image", image_format="png"))
    assert message.to_openai_format() == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,image"}}],
    }

    message = ChatMessage.from_assistant(
        ["this is text", "image_url:images.com/test.jpg", ByteStream.from_base64_image(b"IMAGE")]
    )
    assert message.to_openai_format() == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "this is text"},
            {"type": "image_url", "image_url": {"url": "images.com/test.jpg"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,IMAGE"}},
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


def test_to_dict():
    message = ChatMessage.from_user("content")
    message.meta["some"] = "some"

    assert message.to_dict() == {"content": "content", "role": "user", "name": None, "meta": {"some": "some"}}

    message = ChatMessage.from_assistant("image_url:images.com/test.jpg")
    assert message.to_dict() == {
        "content": "image_url:images.com/test.jpg",
        "role": "assistant",
        "name": None,
        "meta": {},
    }

    message = ChatMessage.from_system(ByteStream(b"bytes", mime_type="image_base64/jpg"))
    assert message.to_dict() == {
        "content": {"data": b"bytes", "mime_type": "image_base64/jpg", "meta": {}},
        "role": "system",
        "name": None,
        "meta": {},
    }

    message = ChatMessage.from_user(
        content=["string content", "image_url:images.com/test.jpg", ByteStream(b"bytes", mime_type="image_base64/png")]
    )
    assert message.to_dict() == {
        "content": [
            "string content",
            "image_url:images.com/test.jpg",
            {"data": b"bytes", "mime_type": "image_base64/png", "meta": {}},
        ],
        "role": "user",
        "name": None,
        "meta": {},
    }


def test_from_dict():
    assert ChatMessage.from_dict(data={"content": "text", "role": "user", "name": None}) == ChatMessage(
        content="text", role=ChatRole.USER, name=None, meta={}
    )

    assert ChatMessage.from_dict(
        data={"content": "image_url:images.com", "role": "user", "name": None, "meta": {}}
    ) == ChatMessage(content="image_url:images.com", role=ChatRole.USER, name=None, meta={})

    assert ChatMessage.from_dict(
        data={
            "content": {"data": b"bytes", "mime_type": "image_base64/png", "meta": {}},
            "role": "user",
            "name": None,
            "meta": {},
        }
    ) == ChatMessage(
        content=ByteStream(data=b"bytes", mime_type="image_base64/png"), role=ChatRole.USER, name=None, meta={}
    )
    assert ChatMessage.from_dict(
        data={
            "content": [
                "string content",
                "image_url:images.com/test.jpg",
                {"data": b"bytes", "mime_type": "image_base64/jpg", "meta": {}},
            ],
            "role": "user",
            "name": None,
            "meta": {},
        }
    ) == ChatMessage(
        content=["string content", "image_url:images.com/test.jpg", ByteStream(b"bytes", mime_type="image_base64/jpg")],
        role=ChatRole.USER,
        name=None,
        meta={},
    )


def test_from_dict_with_meta():
    assert ChatMessage.from_dict(
        data={"content": "text", "role": "user", "name": None, "meta": {"something": "something"}}
    ) == ChatMessage(content="text", role=ChatRole("user"), name=None, meta={"something": "something"})


def test_post_init_method():
    message = ChatMessage.from_user("Content")
    assert message.content == "Content"
    assert "__haystack_content_type__" in message.meta
    assert message.meta["__haystack_content_type__"] == ContentType.TEXT

    message = ChatMessage.from_assistant("image_url:image.com/test.jpg")
    assert message.content == "image.com/test.jpg"
    assert "__haystack_content_type__" in message.meta
    assert message.meta["__haystack_content_type__"] == ContentType.IMAGE_URL

    message = ChatMessage.from_system(ByteStream(data=b"content", mime_type="image_base64/jpg"))
    assert message.content == ByteStream(data=b"content", mime_type="image_base64/jpg")
    assert "__haystack_content_type__" in message.meta
    assert message.meta["__haystack_content_type__"] == ContentType.IMAGE_BASE64

    message = ChatMessage.from_system(ByteStream(data=b"content", mime_type="image_base64/png"))
    assert message.content == ByteStream(data=b"content", mime_type="image_base64/png")
    assert "__haystack_content_type__" in message.meta
    assert message.meta["__haystack_content_type__"] == ContentType.IMAGE_BASE64

    message = ChatMessage.from_user(
        [
            "content",
            "image_url:{{url}}",
            ByteStream(b"content", mime_type="image_base64/jpg"),
            ByteStream(b"content", mime_type="image_base64/png"),
        ]
    )
    assert message.content == [
        "content",
        "{{url}}",
        ByteStream(b"content", mime_type="image_base64/jpg"),
        ByteStream(b"content", mime_type="image_base64/png"),
    ]
    assert "__haystack_content_type__" in message.meta
    assert message.meta["__haystack_content_type__"] == [
        ContentType.TEXT,
        ContentType.IMAGE_URL,
        ContentType.IMAGE_BASE64,
        ContentType.IMAGE_BASE64,
    ]


def test_post_init_raises_value_error_if_mime_type_is_none_or_invalid():
    with pytest.raises(ValueError):
        ChatMessage.from_user(ByteStream.from_string("content"))

    with pytest.raises(ValueError):
        ChatMessage.from_user(ByteStream(b"content", mime_type="fails"))

    with pytest.raises(ValueError):
        ChatMessage.from_user(ByteStream(b"content", mime_type="text"))
