# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.evaluators.answer_exact_match import AnswerExactMatchEvaluator
from haystack.components.evaluators.document_map import DocumentMAPEvaluator
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.document_recall import DocumentRecallEvaluator, RecallMode


@pytest.mark.parametrize(
    "evaluator",
    [
        DocumentMAPEvaluator(),
        DocumentMRREvaluator(),
        DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT),
        DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT),
    ],
)
def test_document_evaluators_reject_top_level_empty_input(evaluator):
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[])

    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[[Document(content="x")]])

    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[[Document(content="x")]], retrieved_documents=[])


def test_document_evaluators_still_score_a_question_with_no_documents():
    ground_truth = [[]]
    retrieved = [[Document(content="x")]]
    for evaluator in (DocumentMAPEvaluator(), DocumentMRREvaluator(), DocumentRecallEvaluator()):
        result = evaluator.run(ground_truth_documents=ground_truth, retrieved_documents=retrieved)
        assert result["score"] == 0.0
        assert result["individual_scores"] == [0.0]


def test_answer_exact_match_rejects_top_level_empty_input():
    evaluator = AnswerExactMatchEvaluator()
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_answers=[], predicted_answers=[])

    result = evaluator.run(ground_truth_answers=["Berlin"], predicted_answers=["Berlin"])
    assert result == {"individual_scores": [1], "score": 1.0}
