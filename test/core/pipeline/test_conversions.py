from typing import Any, Optional

import pytest

from haystack.core.component import component
from haystack.core.component.types import Variadic
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
class IntListInput:
    @component.output_types(values=list[int])
    def run(self, values: list[int]) -> dict[str, Any]:
        return {"values": values}

@component
class IntListOutput:
    @component.output_types(values=list[int])
    def run(self) -> dict[str, Any]:
        return {"values": [1, 2, 3]}

@component
class IntInput:
    @component.output_types(value=int)
    def run(self, value: int) -> dict[str, Any]:
        return {"value": value}

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

    def test_item_to_list(self):
        pipe = Pipeline()
        pipe.add_component("src", IntOutput())
        pipe.add_component("dest", IntListInput())
        pipe.connect("src.value", "dest.values")

        results = pipe.run({})
        assert results["dest"]["values"] == [10]

    def test_list_to_item(self):
        pipe = Pipeline()
        pipe.add_component("src", IntListOutput())
        pipe.add_component("dest", IntInput())
        pipe.connect("src.values", "dest.value")

        results = pipe.run({})
        assert results["dest"]["value"] == 1

    def test_list_to_item_empty_list_fails_if_not_optional(self):
        @component
        class EmptyListOutput:
            @component.output_types(values=list[int])
            def run(self) -> dict[str, Any]:
                return {"values": []}

        pipe = Pipeline()
        pipe.add_component("src", EmptyListOutput())
        pipe.add_component("dest", IntInput())
        pipe.connect("src.values", "dest.value")

        # Should raise ValueError at runtime because dest.value is int (not Optional[int])
        with pytest.raises(ValueError, match="Cannot convert empty list"):
            pipe.run({})

    def test_list_to_item_empty_list_works_if_optional(self):
        @component
        class EmptyListOutput:
            @component.output_types(values=list[int])
            def run(self) -> dict[str, Any]:
                return {"values": []}

        @component
        class OptionalIntInput:
            @component.output_types(value=Optional[int])
            def run(self, value: int | None = None):
                return {"value": value}

        pipe = Pipeline()
        pipe.add_component("src", EmptyListOutput())
        pipe.add_component("dest", OptionalIntInput())
        pipe.connect("src.values", "dest.value")

        results = pipe.run({})
        assert results["dest"]["value"] is None

    def test_incompatible_types_still_fail(self):
        pipe = Pipeline()
        pipe.add_component("src", IntOutput())
        pipe.add_component("dest", StringInput())
        with pytest.raises(PipelineConnectError):
            pipe.connect("src.value", "dest.text")

    def test_none_value_conversion(self):
        @component
        class NoneOutput:
            @component.output_types(value=int)
            def run(self):
                return {"value": None}

        pipe = Pipeline()
        pipe.add_component("src", NoneOutput())
        pipe.add_component("dest", IntListInput())
        pipe.connect("src.value", "dest.values")

        # Connection should be valid, and runtime should handle None
        results = pipe.run({})
        assert results["dest"]["values"] == [None]

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
        with pytest.raises(ValueError, match="result is None"):
            pipe.run({})

    def test_variadic_socket_with_conversion(self):
        @component
        class VariadicListInput:
            @component.output_types(values=list[list[int]])
            def run(self, values: Variadic[list[int]]):
                return {"values": list(values)}

        pipe = Pipeline()
        pipe.add_component("src1", IntOutput()) # Outputs int, receiver expects List[int]
        pipe.add_component("dest", VariadicListInput())
        pipe.connect("src1.value", "dest.values")

        results = pipe.run({})
        # Variadic inputs are collected into a list.
        # src1.value (10) converts to [10].
        # Variadic collects it into [[10]].
        assert results["dest"]["values"] == [[10]]

    def test_list_of_list_to_list_conversion(self):
        @component
        class ListOfListOutput:
            @component.output_types(values=list[list[int]])
            def run(self):
                return {"values": [[1, 2], [3, 4]]}

        pipe = Pipeline()
        pipe.add_component("src", ListOfListOutput())
        pipe.add_component("dest", IntListInput())
        pipe.connect("src.values", "dest.values")

        results = pipe.run({})
        # Should take the first element: [1, 2]
        assert results["dest"]["values"] == [1, 2]

    def test_composite_conversion_list_chatmessage_to_str(self):
        pipe = Pipeline()
        pipe.add_component("src", ChatMessageOutput()) # Outputs ChatMessage
        @component
        class ChatMessageListOutput:
            @component.output_types(messages=list[ChatMessage])
            def run(self):
                return {"messages": [ChatMessage.from_assistant("Hello")]}

        pipe = Pipeline()
        pipe.add_component("src", ChatMessageListOutput())
        pipe.add_component("dest", StringInput())
        # Should work implicitly and use composite conversion
        pipe.connect("src", "dest")

        results = pipe.run({})
        assert results["dest"]["text"] == "Hello"

    def test_composite_conversion_str_to_list_chatmessage(self):
        pipe = Pipeline()
        pipe.add_component("src", StringOutput())
        pipe.add_component("dest", ChatMessageInput())
        @component
        class ChatMessageListInput:
            @component.output_types(messages=list[ChatMessage])
            def run(self, messages: list[ChatMessage]):
                return {"messages": messages}

        pipe = Pipeline()
        pipe.add_component("src", StringOutput())
        pipe.add_component("dest", ChatMessageListInput())
        # Should work implicitly: str -> ChatMessage -> List[ChatMessage]
        pipe.connect("src", "dest")

        results = pipe.run({})
        assert len(results["dest"]["messages"]) == 1
        assert results["dest"]["messages"][0].text == "Hello"

    def test_list_to_item_not_subscriptable_at_runtime(self):
        @component
        class LiarListOutput:
            @component.output_types(values=list[int])
            def run(self):
                # Claims to return List[int], but returns int
                return {"values": 10}

        pipe = Pipeline()
        pipe.add_component("src", LiarListOutput())
        pipe.add_component("dest", IntInput())
        pipe.connect("src.values", "dest.value")

        # Should raise ValueError because the runtime value (10) is not subscriptable
        # and therefore cannot be converted to a single item for a non-optional input.
        with pytest.raises(ValueError, match="Cannot convert empty list"):
            pipe.run({})

    def test_priority_matching_strict_over_convertible(self):
        @component
        class MultiInput:
            @component.output_types(res=str)
            def run(self, strict_in: list[ChatMessage], conv_in: str):
                return {"res": "ok"}

        pipe = Pipeline()
        @component
        class ChatMessageListOutput:
            @component.output_types(messages=list[ChatMessage])
            def run(self):
                return {"messages": [ChatMessage.from_assistant("Hello")]}

        pipe.add_component("src", ChatMessageListOutput())
        pipe.add_component("dest", MultiInput())

        # List[ChatMessage] can connect to both strict_in (Strict) and conv_in (Convertible via conversion)
        # It should pick strict_in automatically.
        pipe.connect("src", "dest")

        # Verify the connection
        edges = list(pipe.graph.edges(data=True))
        # Find the edge to dest
        dest_edges = [e for e in edges if e[1] == "dest"]
        assert len(dest_edges) == 1
        assert dest_edges[0][2]["to_socket"].name == "strict_in"
