# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, default_from_dict
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
        _ = evaluator.run(ground_truth_documents=[[Document(content="France")]], retrieved_documents=[])


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


def test_to_dict_default():
    evaluator = DocumentNDCGEvaluator()
    data = evaluator.to_dict()
    assert data == {
        "type": "haystack.components.evaluators.document_ndcg.DocumentNDCGEvaluator",
        "init_parameters": {"document_comparison_field": "content"},
    }


def test_to_dict_custom_field():
    evaluator = DocumentNDCGEvaluator(document_comparison_field="id")
    data = evaluator.to_dict()
    assert data == {
        "type": "haystack.components.evaluators.document_ndcg.DocumentNDCGEvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }


def test_from_dict():
    data = {
        "type": "haystack.components.evaluators.document_ndcg.DocumentNDCGEvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }
    evaluator = default_from_dict(DocumentNDCGEvaluator, data)
    assert evaluator.document_comparison_field == "id"


def test_run_with_id_comparison():
    # Documents with same content but different IDs — id comparison
    # must match on id, not content
    evaluator = DocumentNDCGEvaluator(document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[[Document(id="doc1", content="France"), Document(id="doc2", content="Paris")]],
        retrieved_documents=[
            [
                Document(id="doc1", content="different text"),
                Document(id="doc3", content="Germany"),
                Document(id="doc2", content="also different"),
            ]
        ],
    )
    assert result["individual_scores"][0] == pytest.approx(0.9197, abs=1e-4)
    assert result["score"] == pytest.approx(0.9197, abs=1e-4)


def test_run_with_id_comparison_no_match():
    evaluator = DocumentNDCGEvaluator(document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[[Document(id="doc1", content="France")]],
        retrieved_documents=[[Document(id="doc99", content="France")]],
    )
    # Same content, different ID — should NOT match when comparing by id
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_run_with_meta_comparison():
    evaluator = DocumentNDCGEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France", meta={"file_id": "f1"}), Document(content="Paris", meta={"file_id": "f2"})]
        ],
        retrieved_documents=[
            [
                Document(content="different", meta={"file_id": "f1"}),
                Document(content="irrelevant", meta={"file_id": "f99"}),
                Document(content="also different", meta={"file_id": "f2"}),
            ]
        ],
    )
    assert result["individual_scores"][0] == pytest.approx(0.9197, abs=1e-4)
    assert result["score"] == pytest.approx(0.9197, abs=1e-4)


def test_run_with_nested_meta_comparison():
    evaluator = DocumentNDCGEvaluator(document_comparison_field="meta.source.url")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="x", meta={"source": {"url": "https://a.com"}})]],
        retrieved_documents=[[Document(content="z", meta={"source": {"url": "https://a.com"}})]],
    )
    assert result["individual_scores"] == [1.0]
    assert result["score"] == 1.0


def test_run_with_meta_missing_key_treated_as_no_match():
    # Documents missing the meta key should not match anything
    evaluator = DocumentNDCGEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France", meta={"file_id": "f1"})]],
        retrieved_documents=[[Document(content="France", meta={})]],
    )
    assert result["individual_scores"] == [0.0]
    assert result["score"] == 0.0


def test_run_with_id_comparison_with_scores():
    # Verify that relevance scores are honoured when comparing by id
    evaluator = DocumentNDCGEvaluator(document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[
            [
                Document(id="doc1", content="foo", score=3),
                Document(id="doc2", content="bar", score=2),
                Document(id="doc3", content="baz", score=3),
                Document(id="doc6", content="qux", score=2),
                Document(id="doc7", content="quux", score=3),
                Document(id="doc8", content="corge", score=2),
            ]
        ],
        retrieved_documents=[
            [
                Document(id="doc1", content="x"),
                Document(id="doc2", content="y"),
                Document(id="doc3", content="z"),
                Document(id="doc4", content="w"),
                Document(id="doc5", content="v"),
            ]
        ],
    )
    assert result["individual_scores"][0] == pytest.approx(0.6592, abs=1e-4)
    assert result["score"] == pytest.approx(0.6592, abs=1e-4)


def test_unsupported_comparison_field_raises():
    evaluator = DocumentNDCGEvaluator(document_comparison_field="embedding")
    with pytest.raises(ValueError, match="Unsupported document_comparison_field"):
        evaluator.run(
            ground_truth_documents=[[Document(content="France")]], retrieved_documents=[[Document(content="France")]]
        )


def test_run_with_meta_missing_key_can_still_reach_perfect_ndcg():
    """
    Regression test for the IDCG/DCG inflation bug: ground truth documents that
    cannot be matched (missing the configured meta key) must be excluded from
    IDCG too, otherwise NDCG can never reach 1.0 even for a perfect retrieval.
    """
    evaluator = DocumentNDCGEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[
            [
                Document(content="France", meta={"file_id": "f1"}),
                Document(content="unmatchable", meta={}),  # no file_id -> cannot be matched
            ]
        ],
        retrieved_documents=[[Document(content="France", meta={"file_id": "f1"})]],
    )
    # Perfect retrieval of the one matchable document should yield NDCG of exactly 1.0
    assert result["individual_scores"] == [1.0]
    assert result["score"] == 1.0
