import logging

from haystack import GeneratedAnswer
from haystack.components.routers.answer_joiner import AnswerJoiner


class TestAnswerJoiner:
    def test_init(self):
        joiner = AnswerJoiner()
        assert joiner.top_k is None
        assert joiner.sort_by_score

    def test_init_with_custom_parameters(self):
        joiner = AnswerJoiner(top_k=5, sort_by_score=False)
        assert joiner.top_k == 5
        assert not joiner.sort_by_score

    def test_empty_list(self):
        joiner = AnswerJoiner()
        result = joiner.run([])
        assert result == {"answers": []}

    def test_list_of_empty_lists(self):
        joiner = AnswerJoiner()
        result = joiner.run([[], []])
        assert result == {"answers": []}

    def test_list_with_one_empty_list(self):
        joiner = AnswerJoiner()
        answers = [
            GeneratedAnswer(query="query", documents=[], data="a"),
            GeneratedAnswer(query="query", documents=[], data="b"),
            GeneratedAnswer(query="query", documents=[], data="c"),
        ]
        result = joiner.run([[], answers])
        assert result == {"answers": answers}

    def test_run_with_top_k(self):
        joiner = AnswerJoiner(top_k=6)
        answers_1 = [
            GeneratedAnswer(query="query", documents=[], data="a"),
            GeneratedAnswer(query="query", documents=[], data="b"),
            GeneratedAnswer(query="query", documents=[], data="c"),
        ]
        answers_2 = [
            GeneratedAnswer(query="query", documents=[], data="d"),
            GeneratedAnswer(query="query", documents=[], data="e"),
            GeneratedAnswer(query="query", documents=[], data="f", meta={"key": "value"}),
            GeneratedAnswer(query="query", documents=[], data="g"),
        ]
        output = joiner.run([answers_1, answers_2])
        assert len(output["answers"]) == 6
        assert sorted(answers_1 + answers_2[:-1], key=id) == sorted(output["answers"], key=id)

    def test_run_with_duplicate_documents(self):
        joiner = AnswerJoiner()
        answers_1 = [
            GeneratedAnswer(query="query", documents=[], data="a"),
            GeneratedAnswer(query="query", documents=[], data="b"),
            GeneratedAnswer(query="query", documents=[], data="c"),
        ]
        answers_2 = [
            GeneratedAnswer(query="query", documents=[], data="a"),
            GeneratedAnswer(query="query", documents=[], data="a"),
            GeneratedAnswer(query="query", documents=[], data="f", meta={"key": "value"}),
        ]
        output = joiner.run([answers_1, answers_2])
        assert len(output["answers"]) == 4
        assert sorted(answers_1 + [answers_2[-1]], key=id) == sorted(output["answers"], key=id)

    def test_sort_by_score_without_scores(self, caplog):
        joiner = AnswerJoiner()
        with caplog.at_level(logging.INFO):
            answers = [
                GeneratedAnswer(query="query", documents=[], data="a"),
                GeneratedAnswer(query="query", documents=[], data="b"),
            ]
            output = joiner.run([answers])
            assert "those with score=None were sorted as if they had a score of -infinity" in caplog.text
            assert output["answers"] == answers

    def test_output_answers_not_sorted_by_score(self):
        joiner = AnswerJoiner(sort_by_score=False)
        answers_1 = [GeneratedAnswer(query="query", documents=[], data="a")]
        answers_2 = [GeneratedAnswer(query="query", documents=[], data="d")]
        output = joiner.run([answers_1, answers_2])
        assert output["answers"] == answers_1 + answers_2
