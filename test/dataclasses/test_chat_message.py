# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import json

from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall, ToolCallResult, TextContent


def test_tool_call_init():
    tc = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    assert tc.id == "123"
    assert tc.tool_name == "mytool"
    assert tc.arguments == {"a": 1}


def test_tool_call_result_init():
    tcr = ToolCallResult(result="result", origin=ToolCall(id="123", tool_name="mytool", arguments={"a": 1}), error=True)
    assert tcr.result == "result"
    assert tcr.origin == ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    assert tcr.error


def test_text_content_init():
    tc = TextContent(text="Hello")
    assert tc.text == "Hello"


def test_from_assistant_with_valid_content():
    text = "Hello, how can I assist you?"
    message = ChatMessage.from_assistant(text)

    assert message.role == ChatRole.ASSISTANT
    assert message._content == [TextContent(text)]
    assert message.name is None

    assert message.text == text
    assert message.texts == [text]

    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result


def test_from_assistant_with_tool_calls():
    tool_calls = [
        ToolCall(id="123", tool_name="mytool", arguments={"a": 1}),
        ToolCall(id="456", tool_name="mytool2", arguments={"b": 2}),
    ]

    message = ChatMessage.from_assistant(tool_calls=tool_calls)

    assert message.role == ChatRole.ASSISTANT
    assert message._content == tool_calls

    assert message.tool_calls == tool_calls
    assert message.tool_call == tool_calls[0]

    assert not message.texts
    assert not message.text
    assert not message.tool_call_results
    assert not message.tool_call_result


def test_from_user_with_valid_content():
    text = "I have a question."
    message = ChatMessage.from_user(text=text)

    assert message.role == ChatRole.USER
    assert message._content == [TextContent(text)]
    assert message.name is None

    assert message.text == text
    assert message.texts == [text]

    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result


def test_from_user_with_name():
    text = "I have a question."
    message = ChatMessage.from_user(text=text, name="John")

    assert message.name == "John"
    assert message.role == ChatRole.USER
    assert message._content == [TextContent(text)]


def test_from_system_with_valid_content():
    text = "I have a question."
    message = ChatMessage.from_system(text=text)

    assert message.role == ChatRole.SYSTEM
    assert message._content == [TextContent(text)]

    assert message.text == text
    assert message.texts == [text]

    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result


def test_from_tool_with_valid_content():
    tool_result = "Tool result"
    origin = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    message = ChatMessage.from_tool(tool_result, origin, error=False)

    tcr = ToolCallResult(result=tool_result, origin=origin, error=False)

    assert message._content == [tcr]
    assert message.role == ChatRole.TOOL

    assert message.tool_call_result == tcr
    assert message.tool_call_results == [tcr]

    assert not message.tool_calls
    assert not message.tool_call
    assert not message.texts
    assert not message.text


def test_multiple_text_segments():
    texts = [TextContent(text="Hello"), TextContent(text="World")]
    message = ChatMessage(_role=ChatRole.USER, _content=texts)

    assert message.texts == ["Hello", "World"]
    assert len(message) == 2


def test_mixed_content():
    content = [TextContent(text="Hello"), ToolCall(id="123", tool_name="mytool", arguments={"a": 1})]

    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=content)

    assert len(message) == 2
    assert message.texts == ["Hello"]
    assert message.text == "Hello"

    assert message.tool_calls == [content[1]]
    assert message.tool_call == content[1]


def test_function_role_removed():
    with pytest.raises(ValueError):
        ChatRole.from_str("function")


def test_from_function_class_method_removed():
    with pytest.raises(AttributeError):
        ChatMessage.from_function("Result of function invocation", "my_function")


def test_serde():
    # the following message is created just for testing purposes and does not make sense in a real use case

    role = ChatRole.ASSISTANT

    text_content = TextContent(text="Hello")
    tool_call = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    tool_call_result = ToolCallResult(result="result", origin=tool_call, error=False)
    meta = {"some": "info"}

    message = ChatMessage(_role=role, _content=[text_content, tool_call, tool_call_result], _meta=meta)

    serialized_message = message.to_dict()
    assert serialized_message == {
        "content": [
            {"text": "Hello"},
            {"tool_call": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}}},
            {
                "tool_call_result": {
                    "result": "result",
                    "error": False,
                    "origin": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}},
                }
            },
        ],
        "role": "assistant",
        "name": None,
        "meta": {"some": "info"},
    }

    deserialized_message = ChatMessage.from_dict(serialized_message)
    assert deserialized_message == message


def test_to_dict_with_invalid_content_type():
    text_content = TextContent(text="Hello")
    invalid_content = "invalid"

    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[text_content, invalid_content])

    with pytest.raises(TypeError):
        message.to_dict()


def test_from_dict_with_invalid_content_type():
    data = {"_role": "assistant", "_content": [{"text": "Hello"}, "invalid"]}
    with pytest.raises(ValueError):
        ChatMessage.from_dict(data)

    data = {"_role": "assistant", "_content": [{"text": "Hello"}, {"invalid": "invalid"}]}
    with pytest.raises(ValueError):
        ChatMessage.from_dict(data)


def test_from_dict_with_pre29_format():
    """Test that we can deserialize messages serialized with pre-2.9.0 format, where the `content` field is a string."""
    serialized_msg_pre_29 = {
        "role": "user",
        "content": "This is a message",
        "name": "some_name",
        "meta": {"some": "info"},
    }
    msg = ChatMessage.from_dict(serialized_msg_pre_29)

    assert msg.role == ChatRole.USER
    assert msg._content == [TextContent(text="This is a message")]
    assert msg.name == "some_name"
    assert msg.meta == {"some": "info"}


def test_from_dict_with_pre29_format_some_fields_missing():
    serialized_msg_pre_29 = {"role": "user", "content": "This is a message"}
    msg = ChatMessage.from_dict(serialized_msg_pre_29)

    assert msg.role == ChatRole.USER
    assert msg._content == [TextContent(text="This is a message")]
    assert msg.name is None
    assert msg.meta == {}


def test_from_dict_with_pre212_format():
    """
    Test that we can deserialize messages serialized with versions >=2.9.0 and <2.12.0,
    where the serialized message has fields `_role`, `_content`, `_name`, and `_meta`.
    """
    serialized_msg_pre_212 = {
        "_role": "user",
        "_content": [{"text": "This is a message"}],
        "_name": "some_name",
        "_meta": {"some": "info"},
    }
    msg = ChatMessage.from_dict(serialized_msg_pre_212)

    assert msg.role == ChatRole.USER
    assert msg._content == [TextContent(text="This is a message")]
    assert msg.name == "some_name"
    assert msg.meta == {"some": "info"}


def test_from_dict_with_pre212_format_some_fields_missing():
    serialized_msg_pre_212 = {"_role": "user", "_content": [{"text": "This is a message"}]}
    msg = ChatMessage.from_dict(serialized_msg_pre_212)

    assert msg.role == ChatRole.USER
    assert msg._content == [TextContent(text="This is a message")]
    assert msg.name is None
    assert msg.meta == {}


def test_from_dict_some_fields_missing():
    serialized_msg = {"role": "user", "content": [{"text": "This is a message"}]}
    msg = ChatMessage.from_dict(serialized_msg)

    assert msg.role == ChatRole.USER
    assert msg._content == [TextContent(text="This is a message")]
    assert msg.name is None
    assert msg.meta == {}


def test_from_dict_missing_content_field():
    serialized_msg = {"role": "user", "name": "some_name", "meta": {"some": "info"}}
    with pytest.raises(ValueError):
        ChatMessage.from_dict(serialized_msg)


def test_chat_message_content_attribute_removed():
    message = ChatMessage.from_user(text="This is a message")
    with pytest.raises(AttributeError):
        message.content


def test_chat_message_init_parameters_removed():
    with pytest.raises(TypeError):
        ChatMessage(role="irrelevant", content="This is a message")


def test_chat_message_init_content_parameter_type():
    with pytest.raises(TypeError):
        ChatMessage(ChatRole.USER, "This is a message")


def test_to_openai_dict_format():
    message = ChatMessage.from_system("You are good assistant")
    assert message.to_openai_dict_format() == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert message.to_openai_dict_format() == {"role": "user", "content": "I have a question"}

    message = ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})
    assert message.to_openai_dict_format() == {"role": "assistant", "content": "I have an answer"}

    message = ChatMessage.from_assistant(
        tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
    )
    assert message.to_openai_dict_format() == {
        "role": "assistant",
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
        ],
    }

    tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
    message = ChatMessage.from_tool(
        tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
    )
    assert message.to_openai_dict_format() == {"role": "tool", "content": tool_result, "tool_call_id": "123"}

    message = ChatMessage.from_user(text="I have a question", name="John")
    assert message.to_openai_dict_format() == {"role": "user", "content": "I have a question", "name": "John"}

    message = ChatMessage.from_assistant(text="I have an answer", name="Assistant1")
    assert message.to_openai_dict_format() == {"role": "assistant", "content": "I have an answer", "name": "Assistant1"}


def test_to_openai_dict_format_invalid():
    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
    with pytest.raises(ValueError):
        message.to_openai_dict_format()

    message = ChatMessage(
        _role=ChatRole.ASSISTANT,
        _content=[TextContent(text="I have an answer"), TextContent(text="I have another answer")],
    )
    with pytest.raises(ValueError):
        message.to_openai_dict_format()

    tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
    message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
    with pytest.raises(ValueError):
        message.to_openai_dict_format()

    message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
    with pytest.raises(ValueError):
        message.to_openai_dict_format()


def test_from_openai_dict_format_user_message():
    openai_msg = {"role": "user", "content": "Hello, how are you?", "name": "John"}
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "user"
    assert message.text == "Hello, how are you?"
    assert message.name == "John"


def test_from_openai_dict_format_system_message():
    openai_msg = {"role": "system", "content": "You are a helpful assistant"}
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "system"
    assert message.text == "You are a helpful assistant"


def test_from_openai_dict_format_assistant_message_with_content():
    openai_msg = {"role": "assistant", "content": "I can help with that"}
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "assistant"
    assert message.text == "I can help with that"


def test_from_openai_dict_format_assistant_message_with_tool_calls():
    openai_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_123", "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'}}],
    }
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "assistant"
    assert message.text is None
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call.id == "call_123"
    assert tool_call.tool_name == "get_weather"
    assert tool_call.arguments == {"location": "Berlin"}


def test_from_openai_dict_format_tool_message():
    openai_msg = {"role": "tool", "content": "The weather is sunny", "tool_call_id": "call_123"}
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "tool"
    assert message.tool_call_result.result == "The weather is sunny"
    assert message.tool_call_result.origin.id == "call_123"


def test_from_openai_dict_format_tool_without_id():
    openai_msg = {"role": "tool", "content": "The weather is sunny"}
    message = ChatMessage.from_openai_dict_format(openai_msg)
    assert message.role.value == "tool"
    assert message.tool_call_result.result == "The weather is sunny"
    assert message.tool_call_result.origin.id is None


def test_from_openai_dict_format_missing_role():
    with pytest.raises(ValueError):
        ChatMessage.from_openai_dict_format({"content": "test"})


def test_from_openai_dict_format_missing_content():
    with pytest.raises(ValueError):
        ChatMessage.from_openai_dict_format({"role": "user"})


def test_from_openai_dict_format_invalid_tool_calls():
    openai_msg = {"role": "assistant", "tool_calls": [{"invalid": "format"}]}
    with pytest.raises(ValueError):
        ChatMessage.from_openai_dict_format(openai_msg)


def test_from_openai_dict_format_unsupported_role():
    with pytest.raises(ValueError):
        ChatMessage.from_openai_dict_format({"role": "invalid", "content": "test"})


def test_from_openai_dict_format_assistant_missing_content_and_tool_calls():
    with pytest.raises(ValueError):
        ChatMessage.from_openai_dict_format({"role": "assistant", "irrelevant": "irrelevant"})


def test_from_dict_with_invalid_content_improved_error():
    """Test that _deserialize_content provides clear error messages for invalid content formats."""
    # Test with completely invalid content structure
    data = {"role": "user", "content": [{"invalid_key": "some_value"}]}

    with pytest.raises(
        ValueError,
        match=r"Invalid content in ChatMessage.*ChatMessage content must be a list of dictionaries.*'text', 'tool_call', or 'tool_call_result'",
    ):
        ChatMessage.from_dict(data)

    # Test with mixed valid and invalid content
    data = {"role": "assistant", "content": [{"text": "Valid text"}, {"wrong_field": "invalid"}]}

    with pytest.raises(
        ValueError, match=r"Invalid content in ChatMessage.*Valid formats.*text.*tool_call.*tool_call_result"
    ):
        ChatMessage.from_dict(data)
