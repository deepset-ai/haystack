import pytest

from haystack.components.evaluators.answer_f1 import AnswerF1Evaluator


def test_run_with_all_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["Paris"]],
    )

    assert result == {"scores": [1.0, 1.0], "average": 1.0}


def test_run_with_no_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Paris"], ["London"]],
    )

    assert result == {"scores": [0.0, 0.0], "average": 0.0}


def test_run_with_partial_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Berlin"], ["London"]],
    )

    assert result == {"scores": [1.0, 0.0], "average": 0.5}


def test_run_with_different_lengths():
    evaluator = AnswerF1Evaluator()

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
