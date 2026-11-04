# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import pytest

from haystack import Document, GeneratedAnswer
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.dataclasses.chat_message import ChatMessage


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
        assert answers[0].meta["all_messages"] == ["reply1"]
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
        assert answers[0].documents[0].meta["referenced"] is True
        assert answers[0].documents[0].meta["source_index"] == 2

    def test_run_returns_referenced_documents_in_source_order(self):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]", return_only_referenced_documents=True)
        output = component.run(
            query="test query",
            replies=["First [3], then [10], and finally [50]."],
            documents=[Document(content=f"doc{i}") for i in range(1, 51)],
        )
        source_indices = [doc.meta["source_index"] for doc in output["answers"][0].documents]
        assert source_indices == [3, 10, 50]

    def test_run_with_documents_with_reference_pattern_return_all_documents(self):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]", return_only_referenced_documents=False)
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
        assert len(answers[0].documents) == 2
        assert answers[0].documents[0].content == "test doc 1"
        assert answers[0].documents[0].meta["referenced"] is False
        assert answers[0].documents[0].meta["source_index"] == 1
        assert answers[0].documents[1].content == "test doc 2"
        assert answers[0].documents[1].meta["referenced"] is True
        assert answers[0].documents[1].meta["source_index"] == 2

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

    def test_run_with_documents_with_zero_reference(self, caplog):
        # References are 1-based, so [0] is out of range and must not silently
        # resolve to the last document via a negative index.
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        with caplog.at_level(logging.WARNING):
            output = component.run(
                query="test query",
                replies=["Answer: AnswerString[0]"],
                meta=[{}],
                documents=[Document(content="test doc 1"), Document(content="test doc 2")],
            )
        answers = output["answers"]
        assert len(answers) == 1
        assert len(answers[0].documents) == 0
        assert "Document index '0' referenced in Generator output is out of range." in caplog.text

    def test_run_with_zero_reference_in_expanded_range(self, caplog):
        component = AnswerBuilder(reference_pattern="\\[([\\d\\s,-]+)\\]", expand_reference_ranges=True)
        with caplog.at_level(logging.WARNING):
            output = component.run(
                query="test query",
                replies=["Answer: AnswerString[0-1]"],
                meta=[{}],
                documents=[Document(content="test doc 1"), Document(content="test doc 2")],
            )
        answers = output["answers"]
        assert len(answers) == 1
        assert len(answers[0].documents) == 1
        assert answers[0].documents[0].content == "test doc 1"
        assert answers[0].documents[0].meta["referenced"] is True
        assert "Document index '0' referenced in Generator output is out of range." in caplog.text

    @pytest.mark.parametrize("initial_pattern", ["unused pattern", "["])
    def test_run_with_reference_pattern_set_at_runtime(self, initial_pattern):
        component = AnswerBuilder(reference_pattern=initial_pattern)
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

    @pytest.mark.parametrize("reference_pattern", [r"\[\d+\]", r"\[(\d+)\]\[(\d+)\]", "["])
    @pytest.mark.parametrize("at_runtime", [False, True])
    def test_run_rejects_invalid_reference_pattern_without_citations(self, reference_pattern, at_runtime):
        component = AnswerBuilder(reference_pattern=None if at_runtime else reference_pattern)
        kwargs = {"reference_pattern": reference_pattern} if at_runtime else {}
        with pytest.raises(ValueError, match="Reference pattern"):
            component.run(query="query", replies=["No citations."], documents=[Document(content="doc")], **kwargs)

    def test_run_reference_pattern_requires_exactly_one_capture_group(self):
        reference_pattern = r"\[\d+\]"
        message = (
            f"Reference pattern '{reference_pattern}' contains 0 capture groups. "
            "Please specify a pattern with exactly one capture group that matches the document index."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            AnswerBuilder(reference_pattern=reference_pattern).run(query="query", replies=["See [1]."])

    @pytest.mark.parametrize("expand_reference_ranges", [False, True])
    @pytest.mark.parametrize("reference_pattern, reply", [(r"\[(\d+)?\]", "See []."), (r"\[(.*?)\]", "See [abc].")])
    def test_run_rejects_reference_captures_that_are_not_indices(
        self, reference_pattern, reply, expand_reference_ranges
    ):
        component = AnswerBuilder(reference_pattern=reference_pattern, expand_reference_ranges=expand_reference_ranges)
        with pytest.raises(ValueError, match="Reference pattern.*document index"):
            component.run(query="query", replies=[reply], documents=[Document(content="doc")])

    @pytest.mark.parametrize("reference_pattern", [r"\[(\d+)?\]", r"(?:Document\[)(?P<index>\d+)\]"])
    def test_run_accepts_single_capture_reference_patterns(self, reference_pattern):
        component = AnswerBuilder(reference_pattern=reference_pattern)
        output = component.run(
            query="query", replies=["See Document[2]."], documents=[Document(content="one"), Document(content="two")]
        )
        assert [doc.content for doc in output["answers"][0].documents] == ["two"]

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
        _check_metadata_excluding_all_messages(answers[0].meta, message_meta)
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

        _check_metadata_excluding_all_messages(answers[0].meta, message_meta)
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
        _check_metadata_excluding_all_messages(answers[0].meta, message_meta)
        assert "all_messages" in answers[0].meta

        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 1
        assert answers[0].documents[0].content == "test doc 2"
        assert answers[0].meta["all_messages"] == [
            ChatMessage.from_assistant("Answer: AnswerString[2]", meta=message_meta)
        ]

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
        _check_metadata_excluding_all_messages(answers[0].meta, message_meta)
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

    def test_conversation_history_with_last_message_only_false(self):
        """Test behavior with last_message_only=False (default)"""
        # Test with two messages in replies
        component = AnswerBuilder(last_message_only=False)
        replies = [
            ChatMessage.from_user("What is Haystack?"),
            ChatMessage.from_assistant("Haystack is a framework for building LLM applications."),
        ]
        output = component.run(query="test query", replies=replies)

        answers = output["answers"]
        # Should have one answer for each message in replies
        assert len(answers) == 2
        assert answers[0].data == "What is Haystack?"
        assert answers[1].data == "Haystack is a framework for building LLM applications."

        # Check that each answer contains the full conversation history
        for answer in answers:
            assert "all_messages" in answer.meta
            assert len(answer.meta["all_messages"]) == 2
            assert answer.meta["all_messages"] == replies

        # Test with one message in replies
        component = AnswerBuilder(last_message_only=False)
        replies = [ChatMessage.from_user("What is Haystack?")]
        output = component.run(query="test query", replies=replies)

        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "What is Haystack?"
        assert "all_messages" in answers[0].meta
        assert answers[0].meta["all_messages"] == replies
        assert len(answers[0].meta["all_messages"]) == 1

    def test_conversation_history_with_last_message_only_true(self):
        """Test behavior with last_message_only=True"""
        # Test with two messages in replies
        component = AnswerBuilder(last_message_only=True)
        replies = [
            ChatMessage.from_user("What is Haystack?"),
            ChatMessage.from_assistant("Haystack is a framework for building LLM applications."),
        ]
        output = component.run(query="test query", replies=replies)

        answers = output["answers"]
        # Should only have one answer - the last message
        assert len(answers) == 1
        assert answers[0].data == "Haystack is a framework for building LLM applications."

        # Check that the answer contains the full conversation history
        assert "all_messages" in answers[0].meta
        assert len(answers[0].meta["all_messages"]) == 2
        assert answers[0].meta["all_messages"] == replies

        # Test with one message in replies
        component = AnswerBuilder(last_message_only=True)
        replies = [ChatMessage.from_user("What is Haystack?")]
        output = component.run(query="test query", replies=replies)

        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "What is Haystack?"
        assert "all_messages" in answers[0].meta
        assert answers[0].meta["all_messages"] == replies
        assert len(answers[0].meta["all_messages"]) == 1

    def test_run_does_not_mutate_input_documents_meta(self):
        doc = Document(content="Paris is the capital of France.", meta={"source": "wiki"})
        component = AnswerBuilder()
        component.run(query="Capital of France?", replies=["Paris."], documents=[doc])
        assert doc.meta == {"source": "wiki"}

    def test_run_does_not_mutate_input_documents_meta_with_reference_pattern(self):
        doc = Document(content="Paris is the capital of France.", meta={"source": "wiki"})
        component = AnswerBuilder(reference_pattern=r"\[(\d+)\]", return_only_referenced_documents=False)
        component.run(query="Capital?", replies=["Paris [1]."], documents=[doc])
        assert doc.meta == {"source": "wiki"}

    def test_run_does_not_mutate_document_with_empty_meta(self):
        doc = Document(content="Paris is the capital of France.")
        component = AnswerBuilder()
        component.run(query="Capital of France?", replies=["Paris."], documents=[doc])
        assert doc.meta == {}

    def test_run_expands_reference_ranges_when_enabled(self):
        docs = [Document(content=f"doc {i}") for i in range(1, 11)]
        component = AnswerBuilder(
            reference_pattern=r"\[(\d+)\]", expand_reference_ranges=True, return_only_referenced_documents=False
        )
        output = component.run(query="test query", replies=["Answer citing sources [6-10]."], documents=docs)
        answer = output["answers"][0]
        referenced_docs = [doc for doc in answer.documents if doc.meta["referenced"]]
        assert [doc.meta["source_index"] for doc in referenced_docs] == [6, 7, 8, 9, 10]

    def test_run_expands_comma_separated_reference_ranges(self):
        docs = [Document(content=f"doc {i}") for i in range(1, 11)]
        component = AnswerBuilder(
            reference_pattern=r"\[(\d+)\]", expand_reference_ranges=True, return_only_referenced_documents=False
        )
        output = component.run(query="test query", replies=["Answer citing sources [1-3,7-9]."], documents=docs)
        answer = output["answers"][0]
        referenced_docs = [doc for doc in answer.documents if doc.meta["referenced"]]
        assert [doc.meta["source_index"] for doc in referenced_docs] == [1, 2, 3, 7, 8, 9]

    def test_run_ignores_invalid_reference_ranges(self):
        docs = [Document(content=f"doc {i}") for i in range(1, 5)]
        component = AnswerBuilder(
            reference_pattern=r"\[(\d+)\]", expand_reference_ranges=True, return_only_referenced_documents=True
        )
        output = component.run(query="test query", replies=["Answer citing sources [3-1]."], documents=docs)
        assert output["answers"][0].documents == []

    def test_run_clamps_reference_range_to_number_of_documents(self):
        docs = [Document(content=f"doc {i}") for i in range(1, 4)]
        component = AnswerBuilder(
            reference_pattern=r"\[(\d+)\]", expand_reference_ranges=True, return_only_referenced_documents=True
        )
        output = component.run(query="test query", replies=["Answer citing sources [1-100]."], documents=docs)
        referenced_docs = output["answers"][0].documents
        assert [doc.meta["source_index"] for doc in referenced_docs] == [1, 2, 3]

    def test_run_does_not_expand_reference_ranges_by_default(self):
        docs = [Document(content=f"doc {i}") for i in range(1, 11)]
        component = AnswerBuilder(reference_pattern=r"\[(\d+)\]", return_only_referenced_documents=True)
        output = component.run(query="test query", replies=["Answer citing sources [6-10]."], documents=docs)
        assert output["answers"][0].documents == []
