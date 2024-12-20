# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer
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


def test_from_function():
    # check warning is raised
    with pytest.warns():
        message = ChatMessage.from_function("Result of function invocation", "my_function")

    assert message.role == ChatRole.TOOL
    assert message.tool_call_result == ToolCallResult(
        result="Result of function invocation",
        origin=ToolCall(id=None, tool_name="my_function", arguments={}),
        error=False,
    )


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
        "_content": [
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
        "_role": "assistant",
        "_name": None,
        "_meta": {"some": "info"},
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


def test_from_dict_with_legacy_init_parameters():
    with pytest.raises(TypeError):
        ChatMessage.from_dict({"role": "user", "content": "This is a message"})


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


def test_chat_message_function_role_deprecated():
    with pytest.warns(DeprecationWarning):
        ChatMessage(ChatRole.FUNCTION, TextContent("This is a message"))


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


@pytest.mark.integration
def test_apply_chat_templating_on_chat_message():
    messages = [ChatMessage.from_system("You are good assistant"), ChatMessage.from_user("I have a question")]
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    formatted_messages = [m.to_openai_dict_format() for m in messages]
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
    formatted_messages = [m.to_openai_dict_format() for m in messages]
    tokenized_messages = tokenizer.apply_chat_template(
        formatted_messages, chat_template=anthropic_template, tokenize=False
    )
    assert tokenized_messages == "You are good assistant\nHuman: I have a question\nAssistant:"
