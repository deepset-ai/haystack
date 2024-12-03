# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack import Document, GeneratedAnswer
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses.chat_message import ChatMessage, ChatRole


class TestAnswerBuilder:
    def test_run_unmatching_input_len(self):
        component = AnswerBuilder()
        with pytest.raises(ValueError):
            component.run(query="query", replies=["reply1"], meta=[{"test": "meta"}, {"test": "meta2"}])

    def test_run_without_meta(self):
        component = AnswerBuilder()
        output = component.run(query="query", replies=["reply1"])
        answers = output["answers"]
        assert answers[0].data == "reply1"
        assert answers[0].meta == {}
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_meta_is_an_empty_list(self):
        component = AnswerBuilder()
        output = component.run(query="query", replies=["reply1"], meta=[])
        answers = output["answers"]
        assert answers[0].data == "reply1"
        assert answers[0].meta == {}
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_meta(self):
        component = AnswerBuilder()
        output = component.run(query="query", replies=["reply1"], meta=[{"test": "meta"}])
        answers = output["answers"]
        assert answers[0].data == "reply1"
        assert answers[0].meta == {"test": "meta"}
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_without_pattern(self):
        component = AnswerBuilder()
        output = component.run(query="test query", replies=["Answer: AnswerString"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_with_capturing_group(self):
        component = AnswerBuilder(pattern=r"Answer: (.*)")
        output = component.run(query="test query", replies=["Answer: AnswerString"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_without_capturing_group(self):
        component = AnswerBuilder(pattern=r"'.*'")
        output = component.run(query="test query", replies=["Answer: 'AnswerString'"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "'AnswerString'"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_with_more_than_one_capturing_group(self):
        with pytest.raises(ValueError, match="contains multiple capture groups"):
            AnswerBuilder(pattern=r"Answer: (.*), (.*)")

    def test_run_with_pattern_set_at_runtime(self):
        component = AnswerBuilder(pattern="unused pattern")
        output = component.run(query="test query", replies=["Answer: AnswerString"], meta=[{}], pattern=r"Answer: (.*)")
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_documents_without_reference_pattern(self):
        component = AnswerBuilder()
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString"],
            meta=[{}],
            documents=[Document(content="test doc 1"), Document(content="test doc 2")],
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 2
        assert answers[0].documents[0].content == "test doc 1"
        assert answers[0].documents[1].content == "test doc 2"

    def test_run_with_documents_with_reference_pattern(self):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString[2]"],
            meta=[{}],
            documents=[Document(content="test doc 1"), Document(content="test doc 2")],
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[2]"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 1
        assert answers[0].documents[0].content == "test doc 2"

    def test_run_with_documents_with_reference_pattern_and_no_match(self, caplog):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        with caplog.at_level(logging.WARNING):
            output = component.run(
                query="test query",
                replies=["Answer: AnswerString[3]"],
                meta=[{}],
                documents=[Document(content="test doc 1"), Document(content="test doc 2")],
            )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[3]"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 0
        assert "Document index '3' referenced in Generator output is out of range." in caplog.text

    def test_run_with_reference_pattern_set_at_runtime(self):
        component = AnswerBuilder(reference_pattern="unused pattern")
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString[2][3]"],
            meta=[{}],
            documents=[Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")],
            reference_pattern="\\[(\\d+)\\]",
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[2][3]"
        assert answers[0].meta == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 2
        assert answers[0].documents[0].content == "test doc 2"
        assert answers[0].documents[1].content == "test doc 3"

    def test_run_with_chat_message_replies_without_pattern(self):
        component = AnswerBuilder()

        message_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
        }
        replies = [ChatMessage.from_assistant("Answer: AnswerString", meta=message_meta)]

        output = component.run(query="test query", replies=replies, meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        assert answers[0].meta == message_meta
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_chat_message_replies_with_pattern(self):
        component = AnswerBuilder(pattern=r"Answer: (.*)")

        message_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
        }
        replies = [ChatMessage.from_assistant("Answer: AnswerString", meta=message_meta)]

        output = component.run(query="test query", replies=replies, meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == message_meta
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_chat_message_replies_with_documents(self):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        message_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
        }
        replies = [ChatMessage.from_assistant("Answer: AnswerString[2]", meta=message_meta)]

        output = component.run(
            query="test query",
            replies=replies,
            meta=[{}],
            documents=[Document(content="test doc 1"), Document(content="test doc 2")],
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[2]"
        assert answers[0].meta == message_meta
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 1
        assert answers[0].documents[0].content == "test doc 2"

    def test_run_with_chat_message_replies_with_pattern_set_at_runtime(self):
        component = AnswerBuilder(pattern="unused pattern")
        message_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
        }
        replies = [ChatMessage.from_assistant("Answer: AnswerString", meta=message_meta)]

        output = component.run(query="test query", replies=replies, meta=[{}], pattern=r"Answer: (.*)")
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == message_meta
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_chat_message_replies_with_meta_set_at_run_time(self):
        component = AnswerBuilder()
        message_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
        }
        replies = [ChatMessage.from_assistant("AnswerString", meta=message_meta)]

        output = component.run(query="test query", replies=replies, meta=[{"test": "meta"}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
            "test": "meta",
        }
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_chat_message_no_meta_with_meta_set_at_run_time(self):
        component = AnswerBuilder()
        replies = [ChatMessage.from_assistant("AnswerString")]
        output = component.run(query="test query", replies=replies, meta=[{"test": "meta"}])

        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].meta == {"test": "meta"}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)
