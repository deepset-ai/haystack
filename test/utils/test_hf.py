# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack.utils.hf import resolve_hf_device_map, convert_message_to_hf_format
from haystack.utils.device import ComponentDevice
from haystack.dataclasses import ChatMessage, ToolCall, ChatRole, TextContent


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
