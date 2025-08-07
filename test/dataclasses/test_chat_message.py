# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack.dataclasses.chat_message import ChatMessage, ChatRole, TextContent, ToolCall, ToolCallResult
from haystack.dataclasses.image_content import ImageContent


class TestChatRole:
    def test_chat_role_from_str(self):
        assert ChatRole.from_str("user") == ChatRole.USER

        with pytest.raises(ValueError):
            ChatRole.from_str("invalid")

    def test_function_role_removed(self):
        with pytest.raises(ValueError):
            ChatRole.from_str("function")


class TestContentParts:
    def test_tool_call_init(self):
        tc = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
        assert tc.id == "123"
        assert tc.tool_name == "mytool"
        assert tc.arguments == {"a": 1}

    def test_tool_call_to_dict(self):
        tc = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
        assert tc.to_dict() == {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}}

    def test_tool_call_from_dict(self):
        tc = ToolCall.from_dict({"id": "123", "tool_name": "mytool", "arguments": {"a": 1}})
        assert tc.id == "123"
        assert tc.tool_name == "mytool"
        assert tc.arguments == {"a": 1}

    def test_tool_call_result_init(self):
        tcr = ToolCallResult(
            result="result", origin=ToolCall(id="123", tool_name="mytool", arguments={"a": 1}), error=True
        )
        assert tcr.result == "result"
        assert tcr.origin == ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
        assert tcr.error

    def test_tool_call_result_to_dict(self):
        tcr = ToolCallResult(
            result="result", origin=ToolCall(id="123", tool_name="mytool", arguments={"a": 1}), error=True
        )
        assert tcr.to_dict() == {
            "result": "result",
            "origin": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}},
            "error": True,
        }

    def test_tool_call_result_from_dict(self):
        tcr = ToolCallResult.from_dict(
            {"result": "result", "origin": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}}, "error": True}
        )
        assert tcr.result == "result"
        assert tcr.origin == ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
        assert tcr.error

        with pytest.raises(ValueError):
            ToolCallResult.from_dict({"result": "result", "error": False})

    def test_text_content_init(self):
        tc = TextContent(text="Hello")
        assert tc.text == "Hello"

    def test_text_content_to_dict(self):
        tc = TextContent(text="Hello")
        assert tc.to_dict() == {"text": "Hello"}

    def test_text_content_from_dict(self):
        tc = TextContent.from_dict({"text": "Hello"})
        assert tc.text == "Hello"


class TestChatMessage:
    def test_from_assistant_with_valid_content(self):
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
        assert not message.images
        assert not message.image

    def test_from_assistant_with_tool_calls(self):
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
        assert not message.images
        assert not message.image

    def test_from_user_with_valid_content(self):
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
        assert not message.images
        assert not message.image

    def test_from_user_with_name(self):
        text = "I have a question."
        message = ChatMessage.from_user(text=text, name="John")

        assert message.name == "John"
        assert message.role == ChatRole.USER
        assert message._content == [TextContent(text)]

    def test_from_user_fails_if_no_text_or_content_parts(self):
        with pytest.raises(ValueError):
            ChatMessage.from_user()

    def test_from_user_fails_if_text_and_content_parts(self):
        with pytest.raises(ValueError):
            ChatMessage.from_user(text="text", content_parts=[TextContent(text="text")])

    def test_from_user_empty_text(self):
        message = ChatMessage.from_user(text="")
        assert message.role == ChatRole.USER
        assert message._content == [TextContent(text="")]
        assert message.text == ""
        assert message.texts == [""]

    def test_from_user_with_content_parts(self, base64_image_string):
        content_parts = [TextContent(text="text"), ImageContent(base64_image=base64_image_string)]
        message = ChatMessage.from_user(content_parts=content_parts)

        assert message.role == ChatRole.USER
        assert message._content == content_parts

        content_parts = ["text", ImageContent(base64_image=base64_image_string)]
        message = ChatMessage.from_user(content_parts=content_parts)

        assert message.role == ChatRole.USER
        assert message._content == [TextContent(text="text"), ImageContent(base64_image=base64_image_string)]

        content_parts = [ImageContent(base64_image=base64_image_string)]
        message = ChatMessage.from_user(content_parts=content_parts)

        assert message.role == ChatRole.USER
        assert message._content == [ImageContent(base64_image=base64_image_string)]

    def test_from_user_with_content_parts_fails_unsupported_parts(self):
        with pytest.raises(ValueError):
            ChatMessage.from_user(
                content_parts=["text part", ToolCall(id="123", tool_name="mytool", arguments={"a": 1})]
            )

    def test_from_user_with_content_parts_fails_with_empty_parts(self):
        with pytest.raises(ValueError):
            ChatMessage.from_user(content_parts=[])

    def test_from_system_with_valid_content(self):
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
        assert not message.images
        assert not message.image

    def test_from_tool_with_valid_content(self):
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
        assert not message.images
        assert not message.image

    def test_multiple_text_segments(self):
        texts = [TextContent(text="Hello"), TextContent(text="World")]
        message = ChatMessage(_role=ChatRole.USER, _content=texts)

        assert message.texts == ["Hello", "World"]
        assert len(message) == 2

    def test_mixed_content(self):
        content = [TextContent(text="Hello"), ToolCall(id="123", tool_name="mytool", arguments={"a": 1})]

        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=content)

        assert len(message) == 2
        assert message.texts == ["Hello"]
        assert message.text == "Hello"

        assert message.tool_calls == [content[1]]
        assert message.tool_call == content[1]

    def test_from_function_class_method_removed(self):
        with pytest.raises(AttributeError):
            ChatMessage.from_function("Result of function invocation", "my_function")

    def test_serde(self, base64_image_string):
        # the following message is created just for testing purposes and does not make sense in a real use case

        role = ChatRole.ASSISTANT

        text_content = TextContent(text="Hello")
        tool_call = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
        tool_call_result = ToolCallResult(result="result", origin=tool_call, error=False)
        image_content = ImageContent(
            base64_image=base64_image_string,
            mime_type="image/png",
            detail="auto",
            meta={"key": "value"},
            validation=True,
        )
        meta = {"some": "info"}

        message = ChatMessage(
            _role=role, _content=[text_content, tool_call, tool_call_result, image_content], _meta=meta
        )

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
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": "auto",
                        "meta": {"key": "value"},
                        "validation": True,
                    }
                },
            ],
            "role": "assistant",
            "name": None,
            "meta": {"some": "info"},
        }

        deserialized_message = ChatMessage.from_dict(serialized_message)
        assert deserialized_message == message

    def test_to_dict_with_invalid_content_type(self):
        text_content = TextContent(text="Hello")
        invalid_content = "invalid"

        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[text_content, invalid_content])

        with pytest.raises(TypeError):
            message.to_dict()

    def test_from_dict_with_invalid_content_type(self):
        data = {"role": "assistant", "content": [{"text": "Hello"}, "invalid"]}
        with pytest.raises(ValueError, match="Unsupported content part in the serialized ChatMessage"):
            ChatMessage.from_dict(data)

        data = {"role": "assistant", "content": [{"text": "Hello"}, {"invalid": "invalid"}]}
        with pytest.raises(ValueError, match="Unsupported content part in the serialized ChatMessage"):
            ChatMessage.from_dict(data)

    def test_from_dict_with_missing_role(self):
        data = {"content": [{"text": "Hello"}], "meta": {}}

        with pytest.raises(ValueError, match=r"The `role` field is required"):
            ChatMessage.from_dict(data)

    def test_from_dict_with_pre29_format(self):
        """
        Test that we can deserialize messages serialized with pre-2.9.0 format, where the `content` field is a string.
        """
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

    def test_from_dict_with_pre29_format_some_fields_missing(self):
        serialized_msg_pre_29 = {"role": "user", "content": "This is a message"}
        msg = ChatMessage.from_dict(serialized_msg_pre_29)

        assert msg.role == ChatRole.USER
        assert msg._content == [TextContent(text="This is a message")]
        assert msg.name is None
        assert msg.meta == {}

    def test_from_dict_with_pre212_format(self):
        """
        Test for ChatMessage.from_dict

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

    def test_from_dict_with_pre212_format_some_fields_missing(self):
        serialized_msg_pre_212 = {"_role": "user", "_content": [{"text": "This is a message"}]}
        msg = ChatMessage.from_dict(serialized_msg_pre_212)

        assert msg.role == ChatRole.USER
        assert msg._content == [TextContent(text="This is a message")]
        assert msg.name is None
        assert msg.meta == {}

    def test_from_dict_some_fields_missing(self):
        serialized_msg = {"role": "user", "content": [{"text": "This is a message"}]}
        msg = ChatMessage.from_dict(serialized_msg)

        assert msg.role == ChatRole.USER
        assert msg._content == [TextContent(text="This is a message")]
        assert msg.name is None
        assert msg.meta == {}

    def test_from_dict_missing_content_field(self):
        serialized_msg = {"role": "user", "name": "some_name", "meta": {"some": "info"}}
        with pytest.raises(ValueError):
            ChatMessage.from_dict(serialized_msg)

    def test_chat_message_content_attribute_removed(self):
        message = ChatMessage.from_user(text="This is a message")
        with pytest.raises(AttributeError):
            message.content

    def test_chat_message_init_parameters_removed(self):
        with pytest.raises(TypeError):
            ChatMessage(role="irrelevant", content="This is a message")


class TestToOpenaiDictFormat:
    def test_to_openai_dict_format_system_message(self):
        message = ChatMessage.from_system("You are good assistant")
        assert message.to_openai_dict_format() == {"role": "system", "content": "You are good assistant"}

    def test_to_openai_dict_format_user_message(self):
        message = ChatMessage.from_user("I have a question")
        assert message.to_openai_dict_format() == {"role": "user", "content": "I have a question"}

    def test_to_openai_dict_format_multimodal_user_message(self, base64_image_string):
        message = ChatMessage.from_user(
            content_parts=[
                TextContent("I have a question"),
                ImageContent(base64_image=base64_image_string, detail="auto"),
            ]
        )
        assert message.to_openai_dict_format() == {
            "role": "user",
            "content": [
                {"type": "text", "text": "I have a question"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image_string}", "detail": "auto"},
                },
            ],
        }

        # image content only should be supported as well
        message = ChatMessage.from_user(content_parts=[ImageContent(base64_image=base64_image_string, detail="auto")])
        assert message.to_openai_dict_format() == {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image_string}", "detail": "auto"},
                }
            ],
        }

    def test_to_openai_dict_format_assistant_message(self):
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

    def test_to_openai_dict_format_tool_message(self):
        tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
        message = ChatMessage.from_tool(
            tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        )
        assert message.to_openai_dict_format() == {"role": "tool", "content": tool_result, "tool_call_id": "123"}

    def test_to_openai_dict_format_with_name(self):
        message = ChatMessage.from_user(text="I have a question", name="John")
        assert message.to_openai_dict_format() == {"role": "user", "content": "I have a question", "name": "John"}

        message = ChatMessage.from_assistant(text="I have an answer", name="Assistant1")
        assert message.to_openai_dict_format() == {
            "role": "assistant",
            "content": "I have an answer",
            "name": "Assistant1",
        }

    def test_to_openai_dict_format_invalid(self):
        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
        with pytest.raises(ValueError):
            message.to_openai_dict_format()

        message = ChatMessage(
            _role=ChatRole.USER,
            _content=[
                TextContent(text="I have an answer"),
                ToolCallResult(
                    result="I have another answer",
                    origin=ToolCall(id="123", tool_name="mytool", arguments={"a": 1}),
                    error=False,
                ),
            ],
        )
        with pytest.raises(ValueError):
            message.to_openai_dict_format()

    def test_to_openai_dict_format_require_tool_call_ids(self):
        tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        with pytest.raises(ValueError):
            message.to_openai_dict_format(require_tool_call_ids=True)

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        with pytest.raises(ValueError):
            message.to_openai_dict_format(require_tool_call_ids=True)

    def test_to_openai_dict_format_require_tool_call_ids_false(self):
        tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        openai_msg = message.to_openai_dict_format(require_tool_call_ids=False)

        assert openai_msg == {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}],
        }

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        openai_msg = message.to_openai_dict_format(require_tool_call_ids=False)
        assert openai_msg == {"role": "tool", "content": "result"}


class TestFromOpenaiDictFormat:
    def test_from_openai_dict_format_user_message(self):
        openai_msg = {"role": "user", "content": "Hello, how are you?", "name": "John"}
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "user"
        assert message.text == "Hello, how are you?"
        assert message.name == "John"

    def test_from_openai_dict_format_system_message(self):
        openai_msg = {"role": "system", "content": "You are a helpful assistant"}
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "system"
        assert message.text == "You are a helpful assistant"

    def test_from_openai_dict_format_assistant_message_with_content(self):
        openai_msg = {"role": "assistant", "content": "I can help with that"}
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "assistant"
        assert message.text == "I can help with that"

    def test_from_openai_dict_format_assistant_message_with_tool_calls(self):
        openai_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_123", "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'}}
            ],
        }
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "assistant"
        assert message.text is None
        assert len(message.tool_calls) == 1
        tool_call = message.tool_calls[0]
        assert tool_call.id == "call_123"
        assert tool_call.tool_name == "get_weather"
        assert tool_call.arguments == {"location": "Berlin"}

    def test_from_openai_dict_format_tool_message(self):
        openai_msg = {"role": "tool", "content": "The weather is sunny", "tool_call_id": "call_123"}
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "tool"
        assert message.tool_call_result.result == "The weather is sunny"
        assert message.tool_call_result.origin.id == "call_123"

    def test_from_openai_dict_format_tool_without_id(self):
        openai_msg = {"role": "tool", "content": "The weather is sunny"}
        message = ChatMessage.from_openai_dict_format(openai_msg)
        assert message.role.value == "tool"
        assert message.tool_call_result.result == "The weather is sunny"
        assert message.tool_call_result.origin.id is None

    def test_from_openai_dict_format_missing_role(self):
        with pytest.raises(ValueError):
            ChatMessage.from_openai_dict_format({"content": "test"})

    def test_from_openai_dict_format_missing_content(self):
        with pytest.raises(ValueError):
            ChatMessage.from_openai_dict_format({"role": "user"})

    def test_from_openai_dict_format_invalid_tool_calls(self):
        openai_msg = {"role": "assistant", "tool_calls": [{"invalid": "format"}]}
        with pytest.raises(ValueError):
            ChatMessage.from_openai_dict_format(openai_msg)

    def test_from_openai_dict_format_unsupported_role(self):
        with pytest.raises(ValueError):
            ChatMessage.from_openai_dict_format({"role": "invalid", "content": "test"})

    def test_from_openai_dict_format_assistant_missing_content_and_tool_calls(self):
        with pytest.raises(ValueError):
            ChatMessage.from_openai_dict_format({"role": "assistant", "irrelevant": "irrelevant"})
