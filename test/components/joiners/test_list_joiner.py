# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
import pytest

from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.answer import GeneratedAnswer
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners.list_joiner import ListJoiner
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.core.errors import PipelineConnectError
from haystack.utils.auth import Secret


class TestListJoiner:
    def test_init(self):
        joiner = ListJoiner(List[ChatMessage])
        assert isinstance(joiner, ListJoiner)
        assert joiner.list_type_ == List[ChatMessage]

    def test_to_dict_defaults(self):
        joiner = ListJoiner()
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.list_joiner.ListJoiner",
            "init_parameters": {"list_type_": None},
        }

    def test_to_dict_non_default(self):
        joiner = ListJoiner(List[ChatMessage])
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.list_joiner.ListJoiner",
            "init_parameters": {"list_type_": "typing.List[haystack.dataclasses.chat_message.ChatMessage]"},
        }

    def test_from_dict_default(self):
        data = {"type": "haystack.components.joiners.list_joiner.ListJoiner", "init_parameters": {"list_type_": None}}
        list_joiner = ListJoiner.from_dict(data)
        assert isinstance(list_joiner, ListJoiner)
        assert list_joiner.list_type_ is None

    def test_from_dict_non_default(self):
        data = {
            "type": "haystack.components.joiners.list_joiner.ListJoiner",
            "init_parameters": {"list_type_": "typing.List[haystack.dataclasses.chat_message.ChatMessage]"},
        }
        list_joiner = ListJoiner.from_dict(data)
        assert isinstance(list_joiner, ListJoiner)
        assert list_joiner.list_type_ == List[ChatMessage]

    def test_empty_list(self):
        joiner = ListJoiner(List[ChatMessage])
        result = joiner.run([])
        assert result == {"values": []}

    def test_list_of_empty_lists(self):
        joiner = ListJoiner(List[ChatMessage])
        result = joiner.run([[], []])
        assert result == {"values": []}

    def test_single_list_of_chat_messages(self):
        joiner = ListJoiner(List[ChatMessage])
        messages = [ChatMessage.from_user("Hello"), ChatMessage.from_assistant("Hi there")]
        result = joiner.run([messages])
        assert result == {"values": messages}

    def test_multiple_lists_of_chat_messages(self):
        joiner = ListJoiner(List[ChatMessage])
        messages1 = [ChatMessage.from_user("Hello")]
        messages2 = [ChatMessage.from_assistant("Hi there")]
        messages3 = [ChatMessage.from_system("System message")]
        result = joiner.run([messages1, messages2, messages3])
        assert result == {"values": messages1 + messages2 + messages3}

    def test_list_of_generated_answers(self):
        joiner = ListJoiner(List[GeneratedAnswer])
        answers1 = [GeneratedAnswer(query="q1", data="a1", meta={}, documents=[Document(content="d1")])]
        answers2 = [GeneratedAnswer(query="q2", data="a2", meta={}, documents=[Document(content="d2")])]
        result = joiner.run([answers1, answers2])
        assert result == {"values": answers1 + answers2}

    def test_list_two_different_types(self):
        joiner = ListJoiner()
        result = joiner.run([["a", "b"], [1, 2]])
        assert result == {"values": ["a", "b", 1, 2]}

    def test_mixed_empty_and_non_empty_lists(self):
        joiner = ListJoiner(List[ChatMessage])
        messages = [ChatMessage.from_user("Hello")]
        result = joiner.run([messages, [], messages])
        assert result == {"values": messages + messages}

    def test_pipeline_connection_validation(self):
        joiner = ListJoiner()
        llm = OpenAIChatGenerator(model="gpt-4o-mini", api_key=Secret.from_token("test-api-key"))
        pipe = Pipeline()
        pipe.add_component("joiner", joiner)
        pipe.add_component("llm", llm)
        with pytest.raises(PipelineConnectError):
            pipe.connect("joiner.values", "llm.messages")
        assert pipe is not None

    def test_pipeline_connection_validation_list_chatmessage(self):
        joiner = ListJoiner(List[ChatMessage])
        llm = OpenAIChatGenerator(model="gpt-4o-mini", api_key=Secret.from_token("test-api-key"))
        pipe = Pipeline()
        pipe.add_component("joiner", joiner)
        pipe.add_component("llm", llm)
        pipe.connect("joiner", "llm.messages")
        assert pipe is not None

    def test_pipeline_bad_connection(self):
        with pytest.raises(PipelineConnectError):
            joiner = ListJoiner()
            query_embedder = SentenceTransformersTextEmbedder()
            pipe = Pipeline()
            pipe.add_component("joiner", joiner)
            pipe.add_component("query_embedder", query_embedder)
            pipe.connect("joiner.values", "query_embedder.text")

    def test_pipeline_bad_connection_different_list_types(self):
        with pytest.raises(PipelineConnectError):
            joiner = ListJoiner(List[str])
            llm = OpenAIChatGenerator(model="gpt-4o-mini", api_key=Secret.from_token("test-api-key"))
            pipe = Pipeline()
            pipe.add_component("joiner", joiner)
            pipe.add_component("llm", llm)
            pipe.connect("joiner.values", "llm.messages")

    def test_result_two_different_types(self):
        pipe = Pipeline()
        pipe.add_component("answer_builder", AnswerBuilder())
        pipe.add_component("chat_prompt_builder", ChatPromptBuilder())
        pipe.add_component("joiner", ListJoiner())
        pipe.connect("answer_builder", "joiner.values")
        pipe.connect("chat_prompt_builder", "joiner.values")
        result = pipe.run(
            data={
                "answer_builder": {"query": "What is nuclear physics?", "replies": ["This is an answer."]},
                "chat_prompt_builder": {"template": [ChatMessage.from_user("Hello")]},
            }
        )
        assert isinstance(result["joiner"]["values"][0], GeneratedAnswer)
        assert isinstance(result["joiner"]["values"][1], ChatMessage)
