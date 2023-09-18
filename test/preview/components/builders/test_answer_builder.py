import logging

import pytest

from haystack.preview import GeneratedAnswer, Document
from haystack.preview.components.builders.answer_builder import AnswerBuilder


class TestAnswerBuilder:
    @pytest.mark.unit
    def test_to_dict(self):
        component = AnswerBuilder()
        data = component.to_dict()
        assert data == {"type": "AnswerBuilder", "init_parameters": {"pattern": None, "reference_pattern": None}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = AnswerBuilder(pattern="pattern", reference_pattern="reference_pattern")
        data = component.to_dict()
        assert data == {
            "type": "AnswerBuilder",
            "init_parameters": {"pattern": "pattern", "reference_pattern": "reference_pattern"},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "AnswerBuilder",
            "init_parameters": {"pattern": "pattern", "reference_pattern": "reference_pattern"},
        }
        component = AnswerBuilder.from_dict(data)
        assert component.pattern == "pattern"
        assert component.reference_pattern == "reference_pattern"

    @pytest.mark.unit
    def test_run_unmatching_input_len(self):
        component = AnswerBuilder()
        with pytest.raises(ValueError):
            component.run(query="query", replies=["reply1", "reply2"], metadata=[])

    def test_run_without_pattern(self):
        component = AnswerBuilder()
        output = component.run(query="test query", replies=["Answer: AnswerString"], metadata=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_with_capturing_group(self):
        component = AnswerBuilder(pattern=r"Answer: (.*)")
        output = component.run(query="test query", replies=["Answer: AnswerString"], metadata=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_without_capturing_group(self):
        component = AnswerBuilder(pattern=r"'.*'")
        output = component.run(query="test query", replies=["Answer: 'AnswerString'"], metadata=[{}])
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "'AnswerString'"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_pattern_with_more_than_one_capturing_group(self):
        with pytest.raises(ValueError, match="contains multiple capture groups"):
            AnswerBuilder(pattern=r"Answer: (.*), (.*)")

    def test_run_with_pattern_set_at_runtime(self):
        component = AnswerBuilder(pattern="unused pattern")
        output = component.run(
            query="test query", replies=["Answer: AnswerString"], metadata=[{}], pattern=r"Answer: (.*)"
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "AnswerString"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert answers[0].documents == []
        assert isinstance(answers[0], GeneratedAnswer)

    def test_run_with_documents_without_reference_pattern(self):
        component = AnswerBuilder()
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString"],
            metadata=[{}],
            documents=[Document(text="test doc 1"), Document(text="test doc 2")],
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 2
        assert answers[0].documents[0].text == "test doc 1"
        assert answers[0].documents[1].text == "test doc 2"

    def test_run_with_documents_with_reference_pattern(self):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString[2]"],
            metadata=[{}],
            documents=[Document(text="test doc 1"), Document(text="test doc 2")],
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[2]"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 1
        assert answers[0].documents[0].text == "test doc 2"

    def test_run_with_documents_with_reference_pattern_and_no_match(self, caplog):
        component = AnswerBuilder(reference_pattern="\\[(\\d+)\\]")
        with caplog.at_level(logging.WARNING):
            output = component.run(
                query="test query",
                replies=["Answer: AnswerString[3]"],
                metadata=[{}],
                documents=[Document(text="test doc 1"), Document(text="test doc 2")],
            )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[3]"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 0
        assert "Document index '3' referenced in Generator output is out of range." in caplog.text

    def test_run_with_reference_pattern_set_at_runtime(self):
        component = AnswerBuilder(reference_pattern="unused pattern")
        output = component.run(
            query="test query",
            replies=["Answer: AnswerString[2][3]"],
            metadata=[{}],
            documents=[Document(text="test doc 1"), Document(text="test doc 2"), Document(text="test doc 3")],
            reference_pattern="\\[(\\d+)\\]",
        )
        answers = output["answers"]
        assert len(answers) == 1
        assert answers[0].data == "Answer: AnswerString[2][3]"
        assert answers[0].metadata == {}
        assert answers[0].query == "test query"
        assert len(answers[0].documents) == 2
        assert answers[0].documents[0].text == "test doc 2"
        assert answers[0].documents[1].text == "test doc 3"
