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


def test_init_with_string_mode():
    evaluator = DocumentRecallEvaluator(mode="single_hit")
    assert evaluator.mode == RecallMode.SINGLE_HIT

    evaluator = DocumentRecallEvaluator(mode="multi_hit")
    assert evaluator.mode == RecallMode.MULTI_HIT


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


class TestDocumentRecallEvaluatorEmptyDocuments:
    @pytest.fixture
    def evaluator_single_hit(self):
        return DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT)

    @pytest.fixture
    def evaluator_multi_hit(self):
        return DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT)

    def test_run_with_empty_ground_truth(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[]], retrieved_documents=[[Document(content="Berlin")]]
        )
        assert result_single == {"individual_scores": [0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[]], retrieved_documents=[[Document(content="Berlin")]]
        )
        assert result_multi == {"individual_scores": [0.0], "score": 0.0}

    def test_run_with_empty_retrieved(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[Document(content="Berlin")]], retrieved_documents=[[]]
        )
        assert result_single == {"individual_scores": [0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content="Berlin")]], retrieved_documents=[[]]
        )
        assert result_multi == {"individual_scores": [0.0], "score": 0.0}

    def test_run_with_empty_strings(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[Document(content="")]], retrieved_documents=[[Document(content="Berlin")]]
        )
        assert result_single == {"individual_scores": [0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content="")]], retrieved_documents=[[Document(content="Berlin")]]
        )
        assert result_multi == {"individual_scores": [0.0], "score": 0.0}

    def test_run_with_all_empty_strings(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[Document(content=""), Document(content="")]],
            retrieved_documents=[[Document(content=""), Document(content="")]],
        )
        assert result_single == {"individual_scores": [0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content=""), Document(content="")]],
            retrieved_documents=[[Document(content=""), Document(content="")]],
        )
        assert result_multi == {"individual_scores": [0.0], "score": 0.0}

    def test_run_with_multiple_empty_ground_truth(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[], [], []],
            retrieved_documents=[
                [Document(content="Berlin")],
                [Document(content="Paris")],
                [Document(content="London")],
            ],
        )
        assert result_single == {"individual_scores": [0.0, 0.0, 0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[], [], []],
            retrieved_documents=[
                [Document(content="Berlin")],
                [Document(content="Paris")],
                [Document(content="London")],
            ],
        )
        assert result_multi == {"individual_scores": [0.0, 0.0, 0.0], "score": 0.0}

    def test_run_with_mixed_empty_ground_truth(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(
            ground_truth_documents=[[], [Document(content="Paris")], []],
            retrieved_documents=[
                [Document(content="Berlin")],
                [Document(content="Paris")],
                [Document(content="London")],
            ],
        )
        assert result_single == {"individual_scores": [0.0, 1.0, 0.0], "score": 0.3333333333333333}

        result_multi = evaluator_multi_hit.run(
            ground_truth_documents=[[], [Document(content="Paris")], []],
            retrieved_documents=[
                [Document(content="Berlin")],
                [Document(content="Paris")],
                [Document(content="London")],
            ],
        )
        assert result_multi == {"individual_scores": [0.0, 1.0, 0.0], "score": 0.3333333333333333}

    def test_run_with_empty_ground_truth_and_empty_retrieved(self, evaluator_single_hit, evaluator_multi_hit):
        result_single = evaluator_single_hit.run(ground_truth_documents=[[], []], retrieved_documents=[[], []])
        assert result_single == {"individual_scores": [0.0, 0.0], "score": 0.0}

        result_multi = evaluator_multi_hit.run(ground_truth_documents=[[], []], retrieved_documents=[[], []])
        assert result_multi == {"individual_scores": [0.0, 0.0], "score": 0.0}


class TestDocumentRecallEvaluatorComplexScenarios:
    @pytest.fixture
    def evaluator_multi_hit(self):
        return DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT)

    def test_run_with_duplicate_ground_truth(self, evaluator_multi_hit):
        result = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content="Berlin"), Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin")]],
        )
        assert result == {"individual_scores": [1.0], "score": 1.0}

    def test_run_with_duplicate_retrieved(self, evaluator_multi_hit):
        result = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin"), Document(content="Berlin")]],
        )
        assert result == {"individual_scores": [1.0], "score": 1.0}

    def test_run_with_mixed_case_content(self, evaluator_multi_hit):
        result = evaluator_multi_hit.run(
            ground_truth_documents=[[Document(content="Berlin"), Document(content="PARIS")]],
            retrieved_documents=[[Document(content="BERLIN"), Document(content="Paris")]],
        )
        assert result == {"individual_scores": [0.0], "score": 0.0}

    def test_run_with_partial_overlap(self, evaluator_multi_hit):
        result = evaluator_multi_hit.run(
            ground_truth_documents=[
                [Document(content="Berlin"), Document(content="Paris"), Document(content="London")]
            ],
            retrieved_documents=[[Document(content="Berlin"), Document(content="Paris")]],
        )
        assert result == {"individual_scores": [0.6666666666666666], "score": 0.6666666666666666}
