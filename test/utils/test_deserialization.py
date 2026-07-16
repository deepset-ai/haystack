# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import pytest
from pydantic import BaseModel

from haystack import Document, Pipeline, component
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.core.errors import DeserializationError
from haystack.dataclasses import ChatMessage
from haystack.utils.deserialization import coerce_pipeline_inputs, deserialize_component_inplace


class ChatGeneratorWithoutFromDict:
    def to_dict(self):
        return {"type": "test_deserialization.ChatGeneratorWithoutFromDict"}


@component
class MessageListEcho:
    @component.output_types(messages=list[ChatMessage])
    def run(self, messages: list[ChatMessage], top_k: int = 3, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"messages": messages}


@component
class SingleMessageEcho:
    @component.output_types(message=ChatMessage)
    def run(self, message: ChatMessage, query: str = "") -> dict[str, Any]:
        return {"message": message}


@component
class DocumentListEcho:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document] | None = None) -> dict[str, Any]:
        return {"documents": documents or []}


class TestDeserializeComponentInplace:
    def test_deserialize_component_inplace(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_generator = OpenAIChatGenerator()
        data = {"chat_generator": chat_generator.to_dict()}
        deserialize_component_inplace(data)
        assert isinstance(data["chat_generator"], OpenAIChatGenerator)
        assert data["chat_generator"].to_dict() == chat_generator.to_dict()

    def test_missing_key(self):
        data = {"some_key": "some_value"}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_component_is_not_a_dict(self):
        data = {"chat_generator": "not_a_dict"}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_type_key_missing(self):
        data = {"chat_generator": {"some_key": "some_value"}}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_class_not_correctly_imported(self):
        data = {"chat_generator": {"type": "invalid.module.InvalidClass"}}
        with pytest.raises(DeserializationError):
            deserialize_component_inplace(data)

    def test_component_no_from_dict_method(self):
        chat_generator = ChatGeneratorWithoutFromDict()
        data = {"chat_generator": chat_generator.to_dict()}
        deserialize_component_inplace(data)
        assert isinstance(data["chat_generator"], ChatGeneratorWithoutFromDict)


class TestCoercePipelineInputs:
    @pytest.fixture
    def pipeline(self):
        pipe = Pipeline()
        pipe.add_component("messages_echo", MessageListEcho())
        pipe.add_component("single_echo", SingleMessageEcho())
        pipe.add_component("documents_echo", DocumentListEcho())
        return pipe

    def test_nested_format(self, pipeline):
        messages = [ChatMessage.from_user("Hi"), ChatMessage.from_assistant("Hello")]
        data: dict[str, Any] = {"messages_echo": {"messages": [message.to_dict() for message in messages], "top_k": 5}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"messages_echo": {"messages": messages, "top_k": 5}}
        assert isinstance(data["messages_echo"]["messages"][0], dict)

    def test_flat_format(self, pipeline):
        messages = [ChatMessage.from_user("Hi")]
        data = {"messages": [message.to_dict() for message in messages], "top_k": 5}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"messages": messages, "top_k": 5}

    def test_single_message_socket(self, pipeline):
        message = ChatMessage.from_user("Hi")
        data = {"single_echo": {"message": message.to_dict(), "query": "some query"}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"single_echo": {"message": message, "query": "some query"}}

    def test_optional_list_socket(self, pipeline):
        documents = [Document(content="Hello"), Document(content="World")]
        data = {"documents_echo": {"documents": [document.to_dict() for document in documents]}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"documents_echo": {"documents": documents}}

    def test_instances_pass_through(self, pipeline):
        messages = [ChatMessage.from_user("Hi")]
        data = {"messages_echo": {"messages": messages}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced["messages_echo"]["messages"][0] is messages[0]

    def test_mixed_list(self, pipeline):
        message = ChatMessage.from_user("Hi")
        data = {"messages_echo": {"messages": [message, ChatMessage.from_assistant("Hello").to_dict()]}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced["messages_echo"]["messages"] == [message, ChatMessage.from_assistant("Hello")]

    def test_non_coercible_values_untouched(self, pipeline):
        data = {"messages_echo": {"messages": [], "top_k": 5, "config": {"a": 1}}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == data

    def test_unknown_inputs_untouched(self, pipeline):
        message_dict = ChatMessage.from_user("Hi").to_dict()
        data = {"unknown_input": [message_dict], "top_k": 5}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"unknown_input": [message_dict], "top_k": 5}

    def test_unknown_component_untouched(self, pipeline):
        message_dict = ChatMessage.from_user("Hi").to_dict()
        data = {"unknown_component": {"messages": [message_dict]}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"unknown_component": {"messages": [message_dict]}}

    def test_variadic_socket(self):
        pipe = Pipeline()
        pipe.add_component("joiner", BranchJoiner(list[ChatMessage]))
        messages = [ChatMessage.from_user("Hi")]
        data = {"value": [message.to_dict() for message in messages], "unrelated": 1}

        coerced = coerce_pipeline_inputs(pipe, data)

        assert coerced["value"] == messages

    def test_pydantic_model_dump_payload(self, pipeline):
        class Response(BaseModel):
            messages: list[ChatMessage]

        messages = [ChatMessage.from_user("Hi"), ChatMessage.from_assistant("Hello")]
        dumped = Response(messages=messages).model_dump(mode="json")
        data = {"messages_echo": {"messages": dumped["messages"]}}

        coerced = coerce_pipeline_inputs(pipeline, data)

        assert coerced == {"messages_echo": {"messages": messages}}
