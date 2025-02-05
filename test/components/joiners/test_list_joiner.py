# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from haystack import Document
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.answer import GeneratedAnswer
from haystack.components.joiners.list_joiner import ListJoiner


class TestListJoiner:
    def test_init(self):
        joiner = ListJoiner(List[ChatMessage])
        assert isinstance(joiner, ListJoiner)
        assert joiner.list_type_ == List[ChatMessage]

    def test_to_dict(self):
        joiner = ListJoiner(List[ChatMessage])
        data = joiner.to_dict()
        assert data == {
            "type": "haystack.components.joiners.list_joiner.ListJoiner",
            "init_parameters": {"list_type_": "typing.List[haystack.dataclasses.chat_message.ChatMessage]"},
        }

    def test_from_dict(self):
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

    def test_mixed_empty_and_non_empty_lists(self):
        joiner = ListJoiner(List[ChatMessage])
        messages = [ChatMessage.from_user("Hello")]
        result = joiner.run([messages, [], messages])
        assert result == {"values": messages + messages}
