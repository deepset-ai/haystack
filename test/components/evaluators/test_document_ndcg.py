# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.components.evaluators.document_ndcg import DocumentNDCGEvaluator


def test_document_ndcg_evaluator_empty_retrieved():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(ground_truth_documents=[[Document(content="France")]], retrieved_documents=[[]])
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_document_ndcg_evaluator_empty_ground_truth():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(ground_truth_documents=[[]], retrieved_documents=[[Document(content="France")]])
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_document_ndcg_evaluator_without_relevance_scores():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France"), Document(content="Paris")]],
        retrieved_documents=[[Document(content="France"), Document(content="Germany"), Document(content="Paris")]],
    )
    # dcg = 1/log2(2) + 0.3333/log2(4)
    # idcg = 1/log2(2) + 0.5/log2(3)
    assert result["individual_scores"][0] == pytest.approx(0.8869, abs=1e-4)
    assert result["score"] == pytest.approx(0.8869, abs=1e-4)


def test_document_ndcg_evaluator_with_relevance_scores():
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
    # dcg = 3/log2(2) + 2/log2(3) + 3/log2(4)
    # idcg = 8.740
    assert result["individual_scores"][0] == pytest.approx(0.6592, abs=1e-4)
    assert result["score"] == pytest.approx(0.6592, abs=1e-4)


def test_document_ndcg_evaluator_different_lengths():
    evaluator = DocumentNDCGEvaluator()
    with pytest.raises(
        ValueError, match="The length of ground_truth_documents and retrieved_documents must be the same."
    ):
        evaluator.run(ground_truth_documents=[[Document(content="France")]], retrieved_documents=[])


def test_document_ndcg_evaluator_none_content():
    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France")]], retrieved_documents=[[Document(content=None)]]
    )
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_calculate_dcg():
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
    dcg = evaluator._calculate_dcg(gt_docs, ret_docs)
    assert dcg == pytest.approx(6.8611, abs=1e-4)


def test_calculate_dcg_no_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1"), Document(content="doc2")]
    # 1 + 0,33/log2(3)
    ret_docs = [Document(content="doc2"), Document(content="doc3"), Document(content="doc1")]
    dcg = evaluator._calculate_dcg(gt_docs, ret_docs)
    assert dcg == pytest.approx(1.1667, abs=1e-4)


def test_calculate_idcg():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [
        Document(content="doc1", score=3),
        Document(content="doc2", score=3),
        Document(content="doc3", score=2),
        Document(content="doc4", score=3),
        Document(content="doc5", score=2),
        Document(content="doc6", score=2),
    ]
    idcg = evaluator._calculate_idcg(gt_docs)
    assert idcg == pytest.approx(8.7403, abs=1e-4)


def test_calculate_idcg_no_scores():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    idcg = evaluator._calculate_idcg(gt_docs)
    assert idcg == pytest.approx(1.4821, abs=1e-4)


def test_calculate_idcg_empty():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = []
    idcg = evaluator._calculate_idcg(gt_docs)
    assert idcg == 0


def test_calculate_dcg_empty():
    evaluator = DocumentNDCGEvaluator()
    gt_docs = [Document(content="doc1")]
    ret_docs = []
    dcg = evaluator._calculate_dcg(gt_docs, ret_docs)
    assert dcg == 0
