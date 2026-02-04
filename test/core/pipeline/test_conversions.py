from typing import Any, Optional

import pytest

from haystack.core.component import component
from haystack.core.errors import PipelineConnectError
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage, ImageContent


@component
class ChatMessageOutput:
    @component.output_types(message=ChatMessage)
    def run(self) -> dict[str, Any]:
        return {"message": ChatMessage.from_assistant("Hello")}


@component
class StringInput:
    @component.output_types(text=str)
    def run(self, text: str) -> dict[str, Any]:
        return {"text": text}


@component
class StringOutput:
    @component.output_types(text=str)
    def run(self) -> dict[str, Any]:
        return {"text": "Hello"}


@component
class ChatMessageInput:
    @component.output_types(message=ChatMessage)
    def run(self, message: ChatMessage) -> dict[str, Any]:
        return {"message": message}


@component
class IntOutput:
    @component.output_types(value=int)
    def run(self) -> dict[str, Any]:
        return {"value": 10}


@component
class IntInput:
    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, Any]:
        return {"value": value}


@component
class FakeGenerator:
    @component.output_types(replies=list[str])
    def run(self, prompt: str) -> dict[str, Any]:
        return {"replies": ["Hello from Generator"]}


@component
class FakeChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant("Hello from ChatGenerator")]}


@component
class FakeRetriever:
    @component.output_types(documents=list[str])
    def run(self, query: str) -> dict[str, Any]:
        return {"documents": [f"Document about {query}"]}


@component
class FakePromptBuilder:
    @component.output_types(prompt=str)
    def run(self, template: str, query: str) -> dict[str, Any]:
        return {"prompt": template.replace("{{query}}", query)}


@component
class FakeChatPromptBuilder:
    @component.output_types(messages=list[ChatMessage])
    def run(self, query: str) -> dict[str, Any]:
        return {"messages": [ChatMessage.from_user(query)]}


class TestImplicitConversions:
    def test_chat_message_to_str(self):
        pipe = Pipeline()
        pipe.add_component("src", ChatMessageOutput())
        pipe.add_component("dest", StringInput())
        pipe.connect("src.message", "dest.text")

        results = pipe.run({})
        assert results["dest"]["text"] == "Hello"

    def test_str_to_chat_message(self):
        pipe = Pipeline()
        pipe.add_component("src", StringOutput())
        pipe.add_component("dest", ChatMessageInput())
        pipe.connect("src.text", "dest.message")

        results = pipe.run({})
        assert isinstance(results["dest"]["message"], ChatMessage)
        assert results["dest"]["message"].text == "Hello"
        assert results["dest"]["message"].role == "user"

    def test_optional_to_str_conversion_fails(self):
        @component
        class OptionalChatMessageOutput:
            @component.output_types(message=Optional[ChatMessage])
            def run(self):
                return {"message": ChatMessage.from_assistant("Hello")}

        pipe = Pipeline()
        pipe.add_component("src", OptionalChatMessageOutput())
        pipe.add_component("dest", StringInput())
        # Should FAIL because Optional[ChatMessage] -> str is unsafe
        with pytest.raises(PipelineConnectError):
            pipe.connect("src.message", "dest.text")

    def test_str_to_optional_chat_message_conversion(self):
        @component
        class OptionalChatMessageInput:
            @component.output_types(message=Optional[ChatMessage])
            def run(self, message: ChatMessage | None):
                return {"message": message}

        pipe = Pipeline()
        pipe.add_component("src", StringOutput())
        pipe.add_component("dest", OptionalChatMessageInput())
        pipe.connect("src.text", "dest.message")

        results = pipe.run({})
        assert isinstance(results["dest"]["message"], ChatMessage)
        assert results["dest"]["message"].text == "Hello"

    def test_chat_message_no_text_to_str_fails_if_not_optional(self):
        @component
        class ImageOnlyMessageOutput:
            @component.output_types(message=ChatMessage)
            def run(self):
                return {"message": ChatMessage.from_user(content_parts=[ImageContent(base64_image="YWI=")])}

        pipe = Pipeline()
        pipe.add_component("src", ImageOnlyMessageOutput())
        pipe.add_component("dest", StringInput())
        pipe.connect("src.message", "dest.text")

        # Should fail because StringInput expects str, but we produced None (no text)
        with pytest.raises(ValueError, match="Cannot convert ChatMessage to str because it has no text."):
            pipe.run({})
