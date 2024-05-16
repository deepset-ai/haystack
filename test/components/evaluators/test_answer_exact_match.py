# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.components.evaluators import AnswerExactMatchEvaluator


def test_run_with_all_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Berlin", "Paris"])

    assert result == {"individual_scores": [1, 1], "score": 1.0}


def test_run_with_no_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Paris", "London"])

    assert result == {"individual_scores": [0, 0], "score": 0.0}


def test_run_with_partial_matching():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Berlin", "London"])

    assert result == {"individual_scores": [1, 0], "score": 0.5}


def test_run_with_complex_data():
    evaluator = AnswerExactMatchEvaluator()
    result = evaluator.run(
        ground_truth_answers=[
            "France",
            "9th century",
            "9th",
            "classical music",
            "classical",
            "11th century",
            "the 11th",
            "Denmark",
            "Iceland",
            "Norway",
            "10th century",
            "10th",
        ],
        predicted_answers=[
            "France",
            "9th century",
            "10th century",
            "9th",
            "classic music",
            "rock music",
            "dubstep",
            "the 11th",
            "11th century",
            "Denmark, Iceland and Norway",
            "10th century",
            "10th",
        ],
    )
    assert result == {"individual_scores": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], "score": 0.3333333333333333}


def test_run_with_different_lengths():
    evaluator = AnswerExactMatchEvaluator()

    with pytest.raises(ValueError):
        evaluator.run(ground_truth_answers=["Berlin"], predicted_answers=["Berlin", "London"])

    with pytest.raises(ValueError):
        evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Berlin"])
