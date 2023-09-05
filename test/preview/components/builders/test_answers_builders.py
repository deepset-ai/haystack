import logging

import pytest

from haystack.preview import GeneratedAnswer, Document
from haystack.preview.components.builders.answers_builder import AnswersBuilder


class TestAnswersBuilder:
    @pytest.mark.unit
    def test_to_dict(self):
        component = AnswersBuilder()
        data = component.to_dict()
        assert data == {"type": "AnswersBuilder", "init_parameters": {"pattern": None, "reference_pattern": None}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = AnswersBuilder(pattern="pattern", reference_pattern="reference_pattern")
        data = component.to_dict()
        assert data == {
            "type": "AnswersBuilder",
            "init_parameters": {"pattern": "pattern", "reference_pattern": "reference_pattern"},
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "AnswersBuilder",
            "init_parameters": {"pattern": "pattern", "reference_pattern": "reference_pattern"},
        }
        component = AnswersBuilder.from_dict(data)
        assert component.pattern == "pattern"
        assert component.reference_pattern == "reference_pattern"

    @pytest.mark.unit
    def test_run_unmatching_input_len(self):
        component = AnswersBuilder()
        with pytest.raises(ValueError):
            component.run(queries=["query"], replies=[["reply1"], ["reply2"]], metadata=[[]])

    def test_run_without_pattern(self):
        component = AnswersBuilder()
        answers = component.run(queries=["test query"], replies=[["Answer: AnswerString"]], metadata=[[{}]])
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "Answer: AnswerString"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert answers[0][0].documents == []
        assert isinstance(answers[0][0], GeneratedAnswer)

    def test_run_with_pattern_with_capturing_group(self):
        component = AnswersBuilder(pattern=r"Answer: (.*)")
        answers = component.run(queries=["test query"], replies=[["Answer: AnswerString"]], metadata=[[{}]])
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "AnswerString"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert answers[0][0].documents == []
        assert isinstance(answers[0][0], GeneratedAnswer)

    def test_run_with_pattern_without_capturing_group(self):
        component = AnswersBuilder(pattern=r"'.*'")
        answers = component.run(queries=["test query"], replies=[["Answer: 'AnswerString'"]], metadata=[[{}]])
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "'AnswerString'"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert answers[0][0].documents == []
        assert isinstance(answers[0][0], GeneratedAnswer)

    def test_run_with_pattern_with_more_than_one_capturing_group(self):
        with pytest.raises(ValueError, match="contains multiple capture groups"):
            component = AnswersBuilder(pattern=r"Answer: (.*), (.*)")

    def test_run_with_pattern_set_at_runtime(self):
        component = AnswersBuilder(pattern="unused pattern")
        answers = component.run(
            queries=["test query"], replies=[["Answer: AnswerString"]], metadata=[[{}]], pattern=r"Answer: (.*)"
        )
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "AnswerString"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert answers[0][0].documents == []
        assert isinstance(answers[0][0], GeneratedAnswer)

    def test_run_with_documents_without_reference_pattern(self):
        component = AnswersBuilder()
        answers = component.run(
            queries=["test query"],
            replies=[["Answer: AnswerString"]],
            metadata=[[{}]],
            documents=[[Document(content="test doc 1"), Document(content="test doc 2")]],
        )
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "Answer: AnswerString"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert len(answers[0][0].documents) == 2
        assert answers[0][0].documents[0].content == "test doc 1"
        assert answers[0][0].documents[1].content == "test doc 2"

    def test_run_with_documents_with_reference_pattern(self):
        component = AnswersBuilder(reference_pattern="\\[(\\d+)\\]")
        answers = component.run(
            queries=["test query"],
            replies=[["Answer: AnswerString[2]"]],
            metadata=[[{}]],
            documents=[[Document(content="test doc 1"), Document(content="test doc 2")]],
        )
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "Answer: AnswerString[2]"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert len(answers[0][0].documents) == 1
        assert answers[0][0].documents[0].content == "test doc 2"

    def test_run_with_documents_with_reference_pattern_and_no_match(self, caplog):
        component = AnswersBuilder(reference_pattern="\\[(\\d+)\\]")
        with caplog.at_level(logging.WARNING):
            answers = component.run(
                queries=["test query"],
                replies=[["Answer: AnswerString[3]"]],
                metadata=[[{}]],
                documents=[[Document(content="test doc 1"), Document(content="test doc 2")]],
            )
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "Answer: AnswerString[3]"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert len(answers[0][0].documents) == 0
        assert "Document index '3' referenced in Generator output is out of range." in caplog.text

    def test_run_with_reference_pattern_set_at_runtime(self):
        component = AnswersBuilder(reference_pattern="unused pattern")
        answers = component.run(
            queries=["test query"],
            replies=[["Answer: AnswerString[2][3]"]],
            metadata=[[{}]],
            documents=[
                [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
            ],
            reference_pattern="\\[(\\d+)\\]",
        )
        assert len(answers) == 1
        assert len(answers[0]) == 1
        assert answers[0][0].data == "Answer: AnswerString[2][3]"
        assert answers[0][0].metadata == {}
        assert answers[0][0].query == "test query"
        assert len(answers[0][0].documents) == 2
        assert answers[0][0].documents[0].content == "test doc 2"
        assert answers[0][0].documents[1].content == "test doc 3"
