# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import Document
from haystack.components.evaluators.document_map import DocumentMAPEvaluator


def test_run_with_all_matching():
    evaluator = DocumentMAPEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
    )

    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_run_with_no_matching():
    evaluator = DocumentMAPEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
    )

    assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}


def test_run_with_partial_matching():
    evaluator = DocumentMAPEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
    )

    assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}


def test_run_with_complex_data():
    evaluator = DocumentMAPEvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
            [Document(content="classical music"), Document(content="classical")],
            [Document(content="11th century"), Document(content="the 11th")],
            [Document(content="Denmark, Iceland and Norway")],
            [Document(content="10th century"), Document(content="10th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
            [Document(content="classical"), Document(content="rock music"), Document(content="dubstep")],
            [Document(content="11th"), Document(content="the 11th"), Document(content="11th century")],
            [Document(content="Denmark"), Document(content="Norway"), Document(content="Iceland")],
            [
                Document(content="10th century"),
                Document(content="the first half of the 10th century"),
                Document(content="10th"),
                Document(content="10th"),
            ],
        ],
    )
    assert result == {
        "individual_scores": [
            1.0,
            pytest.approx(0.8333333333333333),
            1.0,
            pytest.approx(0.5833333333333333),
            0.0,
            pytest.approx(0.8055555555555555),
        ],
        "score": pytest.approx(0.7037037037037037),
    }


def test_run_with_different_lengths():
    with pytest.raises(ValueError):
        evaluator = DocumentMAPEvaluator()
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )

    with pytest.raises(ValueError):
        evaluator = DocumentMAPEvaluator()
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")]],
        )
