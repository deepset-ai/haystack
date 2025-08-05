# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from haystack.dataclasses import ChatMessage, ChatRole, ImageContent, TextContent, ToolCall
from haystack.utils.device import ComponentDevice
from haystack.utils.hf import convert_message_to_hf_format, resolve_hf_device_map

# Test images (1x1 pixel PNGs for testing)
TEST_IMAGE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
# Second PNG image with different content for testing multiple images
TEST_IMAGE_PNG2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQYV2NgAAIAAAUAAarVyFEAAAAASUVORK5CYII="


def test_resolve_hf_device_map_only_device():
    model_kwargs = resolve_hf_device_map(device=None, model_kwargs={})
    assert model_kwargs["device_map"] == ComponentDevice.resolve_device(None).to_hf()


def test_resolve_hf_device_map_only_device_map():
    model_kwargs = resolve_hf_device_map(device=None, model_kwargs={"device_map": "cpu"})
    assert model_kwargs["device_map"] == "cpu"


def test_resolve_hf_device_map_device_and_device_map(caplog):
    with caplog.at_level(logging.WARNING):
        model_kwargs = resolve_hf_device_map(
            device=ComponentDevice.from_str("cpu"), model_kwargs={"device_map": "cuda:0"}
        )
        assert "The parameters `device` and `device_map` from `model_kwargs` are both provided." in caplog.text
    assert model_kwargs["device_map"] == "cuda:0"


def test_convert_message_to_hf_format():
    message = ChatMessage.from_system("You are good assistant")
    assert convert_message_to_hf_format(message) == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert convert_message_to_hf_format(message) == {"role": "user", "content": "I have a question"}

    message = ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})
    assert convert_message_to_hf_format(message) == {"role": "assistant", "content": "I have an answer"}

    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
    )
    assert convert_message_to_hf_format(message) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}
        ],
    }

    message = ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather", arguments={"city": "Paris"})])
    assert convert_message_to_hf_format(message) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}],
    }

    tool_result = {"weather": "sunny", "temperature": "25"}
    message = ChatMessage.from_tool(
        tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
    )
    assert convert_message_to_hf_format(message) == {"role": "tool", "content": tool_result, "tool_call_id": "123"}

    message = ChatMessage.from_tool(
        tool_result=tool_result, origin=ToolCall(tool_name="weather", arguments={"city": "Paris"})
    )
    assert convert_message_to_hf_format(message) == {"role": "tool", "content": tool_result}


def test_convert_message_to_hf_invalid():
    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
    with pytest.raises(ValueError):
        convert_message_to_hf_format(message)

    message = ChatMessage(
        _role=ChatRole.ASSISTANT,
        _content=[TextContent(text="I have an answer"), TextContent(text="I have another answer")],
    )
    with pytest.raises(ValueError):
        convert_message_to_hf_format(message)


def test_convert_message_to_hf_format_with_images():
    # Test image with text message (image-only not supported by ChatMessage.from_user)
    image = ImageContent(base64_image=TEST_IMAGE_PNG, mime_type="image/png")
    message = ChatMessage.from_user(content_parts=["Analyze this image", image])

    result = convert_message_to_hf_format(message)
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG}"}},
        ],
    }
    assert result == expected


def test_convert_message_to_hf_format_with_text_and_images():
    # Test text + image message
    image = ImageContent(base64_image=TEST_IMAGE_PNG, mime_type="image/png")
    message = ChatMessage.from_user(content_parts=["Describe this image", image])

    result = convert_message_to_hf_format(message)
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG}"}},
        ],
    }
    assert result == expected


def test_convert_message_to_hf_format_with_multiple_images():
    # Test multiple images (both PNG format with different content)
    image1 = ImageContent(base64_image=TEST_IMAGE_PNG, mime_type="image/png")
    image2 = ImageContent(base64_image=TEST_IMAGE_PNG2, mime_type="image/png")
    message = ChatMessage.from_user(content_parts=["Compare these images", image1, image2])

    result = convert_message_to_hf_format(message)
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG2}"}},
        ],
    }
    assert result == expected


def test_convert_message_to_hf_format_image_without_mime_type():
    # Test image without explicit MIME type (should be auto-detected as PNG from the data)
    image = ImageContent(base64_image=TEST_IMAGE_PNG)
    message = ChatMessage.from_user(content_parts=["What is this?", image])

    result = convert_message_to_hf_format(message)
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG}"}},
        ],
    }
    assert result == expected


def test_convert_message_to_hf_format_image_only_direct_construction():
    # Test image-only message using direct ChatMessage construction
    image = ImageContent(base64_image=TEST_IMAGE_PNG, mime_type="image/png")
    message = ChatMessage(_role=ChatRole.USER, _content=[image])

    result = convert_message_to_hf_format(message)
    expected = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_PNG}"}}],
    }
    assert result == expected
