# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.components.evaluators.document_recall import DocumentRecallEvaluator, RecallMode
from haystack.dataclasses import Document
from haystack import default_from_dict


def test_init_with_unknown_mode_string():
    with pytest.raises(ValueError):
        DocumentRecallEvaluator(mode="unknown_mode")


class TestDocumentRecallEvaluatorSingleHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
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
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1, 1, 1, 1, 0, 1], "score": 0.8333333333333334}

    def test_run_with_different_lengths(self, evaluator):
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

    def test_to_dict(self, evaluator):
        data = evaluator.to_dict()
        assert data == {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "single_hit"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "single_hit"},
        }
        new_evaluator = default_from_dict(DocumentRecallEvaluator, data)
        assert new_evaluator.mode == RecallMode.SINGLE_HIT


class TestDocumentRecallEvaluatorMultiHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
        result = evaluator.run(
            ground_truth_documents=[
                [Document(content="France")],
                [Document(content="9th century"), Document(content="9th")],
                [Document(content="classical music"), Document(content="classical")],
                [Document(content="11th century"), Document(content="the 11th")],
                [
                    Document(content="Denmark"),
                    Document(content="Iceland"),
                    Document(content="Norway"),
                    Document(content="Denmark, Iceland and Norway"),
                ],
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
        assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
        assert result == {"individual_scores": [1.0, 1.0, 0.5, 1.0, 0.75, 1.0], "score": 0.875}

    def test_run_with_different_lengths(self, evaluator):
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

    def test_to_dict(self, evaluator):
        data = evaluator.to_dict()
        assert data == {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "multi_hit"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.evaluators.document_recall.DocumentRecallEvaluator",
            "init_parameters": {"mode": "multi_hit"},
        }
        new_evaluator = default_from_dict(DocumentRecallEvaluator, data)
        assert new_evaluator.mode == RecallMode.MULTI_HIT
