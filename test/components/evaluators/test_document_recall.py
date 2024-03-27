import pytest

from haystack.components.evaluators.document_recall import DocumentRecallEvaluator, RecallMode
from haystack.dataclasses import Document


def test_init_with_unknown_mode_string():
    with pytest.raises(ValueError):
        DocumentRecallEvaluator(mode="unknown_mode")


class TestDocumentRecallEvaluatorSingleHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )

        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )

        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )

        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
        result = evaluator.run(
            questions=[
                "In what country is Normandy located?",
                "When was the Latin version of the word Norman first recorded?",
                "What developed in Normandy during the 1100s?",
                "In what century did important classical music developments occur in Normandy?",
                "From which countries did the Norse originate?",
                "What century did the Normans first gain their separate identity?",
            ],
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
        assert result == {"individual_scores": [True, True, True, True, False, True], "score": 0.8333333333333334}

    def test_run_with_different_lengths(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?"],
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?", "What is the capital of France?"],
                ground_truth_documents=[[Document(content="Berlin")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?", "What is the capital of France?"],
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")]],
            )


class TestDocumentRecallEvaluatorMultiHit:
    @pytest.fixture
    def evaluator(self):
        return DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT)

    def test_run_with_all_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        )

        assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}

    def test_run_with_no_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
        )

        assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}

    def test_run_with_partial_matching(self, evaluator):
        result = evaluator.run(
            questions=["What is the capital of Germany?", "What is the capital of France?"],
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )

        assert result == {"individual_scores": [1.0, 0.0], "score": 0.5}

    def test_run_with_complex_data(self, evaluator):
        result = evaluator.run(
            questions=[
                "In what country is Normandy located?",
                "When was the Latin version of the word Norman first recorded?",
                "What developed in Normandy during the 1100s?",
                "In what century did important classical music developments occur in Normandy?",
                "From which countries did the Norse originate?",
                "What century did the Normans first gain their separate identity?",
            ],
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
        assert result == {"individual_scores": [1.0, 1.0, 0.5, 1.0, 0.75, 1.0], "score": 0.875}

    def test_run_with_different_lengths(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?"],
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?", "What is the capital of France?"],
                ground_truth_documents=[[Document(content="Berlin")]],
                retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
            )

        with pytest.raises(ValueError):
            evaluator.run(
                questions=["What is the capital of Germany?", "What is the capital of France?"],
                ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
                retrieved_documents=[[Document(content="Berlin")]],
            )
