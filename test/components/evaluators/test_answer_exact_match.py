import pytest

from haystack.components.evaluators import AnswerExactMatchEvaluator


def test_run_with_all_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["Paris"]],
    )

    assert result["result"] == 1.0


def test_run_with_no_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Paris"], ["London"]],
    )

    assert result["result"] == 0.0


def test_run_with_partial_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["London"]],
    )

    assert result["result"] == 0.5


def test_run_with_different_lengths():
    evaluator = AnswerExactMatchEvaluator()

    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?"],
            ground_truth_answers=[["Berlin"], ["Paris"]],
            predicted_answers=[["Berlin"], ["London"]],
        )

    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_answers=[["Berlin"]],
            predicted_answers=[["Berlin"], ["London"]],
        )

    with pytest.raises(ValueError):
        evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_answers=[["Berlin"], ["Paris"]],
            predicted_answers=[["Berlin"]],
        )
