# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.components.evaluators.document_ndcg import DocumentNDCGEvaluator


def test_run_with_scores():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [
                Document(content="doc1", score=3),
                Document(content="doc2", score=2),
                Document(content="doc3", score=3),
                Document(content="doc6", score=2),
                Document(content="doc7", score=3),
                Document(content="doc8", score=2),
            ]
        ],
        retrieved_documents=[
            [
                Document(content="doc1"),
                Document(content="doc2"),
                Document(content="doc3"),
                Document(content="doc4"),
                Document(content="doc5"),
            ]
        ],
    )
    assert result["individual_scores"][0] == pytest.approx(0.6592, abs=1e-4)
    assert result["score"] == pytest.approx(0.6592, abs=1e-4)


def test_run_without_scores():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France"), Document(content="Paris")]],
        retrieved_documents=[[Document(content="France"), Document(content="Germany"), Document(content="Paris")]],
    )
    assert result["individual_scores"][0] == pytest.approx(0.9197, abs=1e-4)
    assert result["score"] == pytest.approx(0.9197, abs=1e-4)


def test_run_with_multiple_lists_of_docs():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France"), Document(content="Paris")],
            [
                Document(content="doc1", score=3),
                Document(content="doc2", score=2),
                Document(content="doc3", score=3),
                Document(content="doc6", score=2),
                Document(content="doc7", score=3),
                Document(content="doc8", score=2),
            ],
        ],
        retrieved_documents=[
            [Document(content="France"), Document(content="Germany"), Document(content="Paris")],
            [
                Document(content="doc1"),
                Document(content="doc2"),
                Document(content="doc3"),
                Document(content="doc4"),
                Document(content="doc5"),
            ],
        ],
    )
    assert result["individual_scores"][0] == pytest.approx(0.9197, abs=1e-4)
    assert result["individual_scores"][1] == pytest.approx(0.6592, abs=1e-4)
    assert result["score"] == pytest.approx(0.7895, abs=1e-4)


def test_run_with_different_lengths():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(ValueError):
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )
    with pytest.raises(ValueError):
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")]],
        )


def test_run_with_mixed_documents_with_and_without_scores():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(ValueError):
        evaluator.run(
            ground_truth_documents=[[Document(content="France", score=3), Document(content="Paris")]],
            retrieved_documents=[[Document(content="France"), Document(content="Germany"), Document(content="Paris")]],
        )


def test_run_empty_retrieved():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(ground_truth_documents=[[Document(content="France")]], retrieved_documents=[[]])
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_run_empty_ground_truth():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(ground_truth_documents=[[]], retrieved_documents=[[Document(content="France")]])
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_run_empty_retrieved_and_empty_ground_truth():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(ground_truth_documents=[[]], retrieved_documents=[[]])
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_run_no_retrieved():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(ValueError):
        result = evaluator.run(ground_truth_documents=[[Document(content="France")]], retrieved_documents=[])


def test_run_no_ground_truth():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(ValueError):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[[Document(content="France")]])


def test_run_no_retrieved_and_no_ground_truth():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(ValueError):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[])


def test_calculate_dcg_with_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [
        Document(content="doc1", score=3),
        Document(content="doc2", score=2),
        Document(content="doc3", score=3),
        Document(content="doc4", score=0),
        Document(content="doc5", score=1),
        Document(content="doc6", score=2),
    ]
    ret_docs = [
        Document(content="doc1"),
        Document(content="doc2"),
        Document(content="doc3"),
        Document(content="doc4"),
        Document(content="doc5"),
        Document(content="doc6"),
    ]
    dcg = evaluator.calculate_dcg(gt_docs, ret_docs)
    assert dcg == pytest.approx(6.8611, abs=1e-4)


def test_calculate_dcg_without_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1"), Document(content="doc2")]
    ret_docs = [Document(content="doc2"), Document(content="doc3"), Document(content="doc1")]
    dcg = evaluator.calculate_dcg(gt_docs, ret_docs)
    assert dcg == pytest.approx(1.5, abs=1e-4)


def test_calculate_dcg_empty():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1")]
    ret_docs = []
    dcg = evaluator.calculate_dcg(gt_docs, ret_docs)
    assert dcg == 0


def test_calculate_idcg_with_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [
        Document(content="doc1", score=3),
        Document(content="doc2", score=3),
        Document(content="doc3", score=2),
        Document(content="doc4", score=3),
        Document(content="doc5", score=2),
        Document(content="doc6", score=2),
    ]
    idcg = evaluator.calculate_idcg(gt_docs)
    assert idcg == pytest.approx(8.7403, abs=1e-4)


def test_calculate_idcg_without_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    idcg = evaluator.calculate_idcg(gt_docs)
    assert idcg == pytest.approx(2.1309, abs=1e-4)


def test_calculate_idcg_empty():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = []
    idcg = evaluator.calculate_idcg(gt_docs)
    assert idcg == 0
