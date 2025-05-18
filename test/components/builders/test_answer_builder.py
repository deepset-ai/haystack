# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack import Document, GeneratedAnswer
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses.chat_message import ChatMessage, ChatRole


def _check_metadata_excluding_all_messages(actual_meta, expected_meta):
    """Helper function to check metadata while ignoring the all_messages field"""
    actual_without_messages = {k: v for k, v in actual_meta.items() if k != "all_messages"}
    assert actual_without_messages == expected_meta


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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta  # Check that all_messages exists
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_meta_is_an_empty_list(self):
        component = AnswerBuilder()
        output = component.run(query="query", replies=["reply1"], meta=[])
        answers = output["answers"]
        assert answers[0].data == "reply1"
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_meta(self):
        component = AnswerBuilder()
        output = component.run(query="query", replies=["reply1"], meta=[{"test": "meta"}])
        answers = output["answers"]
        assert answers[0].data == "reply1"
        _check_metadata_excluding_all_messages(answers[0].meta, {"test": "meta"})
        assert "all_messages" in answers[0].meta
        assert answers[0].query == "query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_without_pattern(self):
        component = AnswerBuilder()
        output = component.run(query="test query", replies=["Answer: AnswerString"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_with_capturing_group(self):
        component = AnswerBuilder(pattern=r"Answer: (.*)")
        output = component.run(query="test query", replies=["Answer: AnswerString"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_without_capturing_group(self):
        component = AnswerBuilder(pattern=r"'.*'")
        output = component.run(query="test query", replies=["Answer: 'AnswerString'"], meta=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "'AnswerString'"
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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
        _check_metadata_excluding_all_messages(answers[0].meta, {})
        assert "all_messages" in answers[0].meta
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

        # Check metadata excluding all_messages
        expected_meta = message_meta.copy()
        _check_metadata_excluding_all_messages(answers[0].meta, expected_meta)
        assert "all_messages" in answers[0].meta

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

        # Check metadata excluding all_messages
        expected_meta = message_meta.copy()
        _check_metadata_excluding_all_messages(answers[0].meta, expected_meta)
        assert "all_messages" in answers[0].meta

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

        # Check metadata excluding all_messages
        expected_meta = message_meta.copy()
        _check_metadata_excluding_all_messages(answers[0].meta, expected_meta)
        assert "all_messages" in answers[0].meta

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

        # Check metadata excluding all_messages
        expected_meta = message_meta.copy()
        _check_metadata_excluding_all_messages(answers[0].meta, expected_meta)
        assert "all_messages" in answers[0].meta

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

        # Check metadata excluding all_messages
        expected_meta = {
            "model": "gpt-4o-mini",
            "index": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 32, "completion_tokens": 153, "total_tokens": 185},
            "test": "meta",
        }
        _check_metadata_excluding_all_messages(answers[0].meta, expected_meta)
        assert "all_messages" in answers[0].meta

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

        # Check metadata excluding all_messages
        _check_metadata_excluding_all_messages(answers[0].meta, {"test": "meta"})
        assert "all_messages" in answers[0].meta

        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_conversation_history_in_all_messages(self):
        """Test that multiple messages in replies are stored in all_messages."""
        component = AnswerBuilder()
        replies = [
            ChatMessage.from_user("What is Haystack?"),
            ChatMessage.from_assistant("Haystack is a framework for building LLM applications."),
        ]
        output = component.run(query="test query", replies=replies)

        answers = output["answers"]
        assert len(answers) == 2  # One answer for each message in replies

        # Check that each answer contains the full conversation history
        for answer in answers:
            assert "all_messages" in answer.meta
            assert answer.meta["all_messages"] == replies
            assert len(answer.meta["all_messages"]) == 2
