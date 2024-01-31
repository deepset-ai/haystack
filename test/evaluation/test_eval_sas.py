import pytest

from haystack import Pipeline
from haystack.dataclasses import GeneratedAnswer
from haystack.evaluation.eval import EvaluationResult


class TestSAS:
    def create_evaluation_result(self, predictions, labels):
        """
        Creates an evaluation result of a RAG pipeline using the list of predictions and labels for testing the
         Semantic Answer Similarity (SAS) Metric.
        """
        runnable = Pipeline()
        inputs = []
        outputs = [
            {"answer_builder": {"answers": [GeneratedAnswer(data=pred, query="", documents=[], meta={})]}}
            for pred in predictions
        ]
        expected_outputs = [
            {"answer_builder": {"answers": [GeneratedAnswer(data=label, query="", documents=[], meta={})]}}
            for label in labels
        ]
        evaluation_result = EvaluationResult(runnable, inputs, outputs, expected_outputs)
        return evaluation_result

    def test_sas_empty_inputs(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with empty inputs.
        """
        runnable = Pipeline()
        inputs = []
        outputs = [
            {"answer_builder": {"answers": []}},
            {"answer_builder": {"answers": []}},
            {"answer_builder": {"answers": []}},
        ]
        expected_outputs = [
            {"answer_builder": {"answers": []}},
            {"answer_builder": {"answers": []}},
            {"answer_builder": {"answers": []}},
        ]
        evaluation_result = EvaluationResult(runnable, inputs, outputs, expected_outputs)
        # Expecting 0% SAS for empty inputs
        sas_result = evaluation_result._calculate_sas(output_key="answers")

        assert sas_result["sas"] == 0.0
        assert sas_result["scores"] == [0.0]

    def test_calculate_sas_with_different_lengths(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with default parameters.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        evaluation_result = self.create_evaluation_result(predictions, labels)

        with pytest.raises(ValueError, match="The number of predictions and labels must be the same."):
            evaluation_result._calculate_sas(output_key="answers")

    @pytest.mark.integration
    def test_sas_same_inputs(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with default parameters.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluation_result = self.create_evaluation_result(predictions, labels)
        sas_result = evaluation_result._calculate_sas(output_key="answers")

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_single_word(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with single-word inputs.
        """
        predictions = ["A construction budget of US $2.3 billion"]
        labels = ["US $2.3 billion"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        sas_result = evaluation_result._calculate_sas(output_key="answers")

        assert sas_result["sas"] == pytest.approx(0.689089)
        assert sas_result["scores"] == [pytest.approx(0.689089)]

    @pytest.mark.integration
    def test_sas_negative_case(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with deliberately mismatched predictions and labels.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "US $2.3 billion",
            "Paris's cultural magnificence is symbolized by the Eiffel Tower",
            "Japan was transformed into a modernized world power after the Meiji Restoration.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        sas_result = evaluation_result._calculate_sas(output_key="answers")

        assert sas_result["sas"] == pytest.approx(0.8227189)
        assert sas_result["scores"] == [pytest.approx(0.689089), pytest.approx(0.870389), pytest.approx(0.908679)]

    @pytest.mark.integration
    def test_sas_ignore_case(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with ignoring case sensitivity.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 BILLION",
            "The EIFFEL TOWER, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The MEIJI RESTORATION in 1868 transformed Japan into a modernized world power.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # SAS after case ignoring
        sas_result = evaluation_result._calculate_sas(output_key="answers", ignore_case=True)

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_ignore_punctuation(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with ignoring punctuation.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower completed in 1889 symbolizes Paris's cultural magnificence",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # SAS after ignoring punctuation
        sas_result = evaluation_result._calculate_sas(output_key="answers", ignore_punctuation=True)

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_ignore_numbers(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with ignoring numbers.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # SAS after ignoring numbers
        sas_result = evaluation_result._calculate_sas(output_key="answers", ignore_numbers=True)

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_regex_ignore(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with ignoring specific regex patterns.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns
        regex_to_ignore = [r"\d+"]
        sas_result = evaluation_result._calculate_sas(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_multiple_ignore_regex(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with multiple ignoring parameters.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US #10.3 billion",
            "The Eiffel Tower!!, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The **Meiji Restoration**, in 1989, transformed Japan into a modernized world power.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns and punctuation excluding whitespaces
        regex_to_ignore = [r"\d+", r"[^\w\s]"]
        sas_result = evaluation_result._calculate_sas(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_multiple_ignore_combination(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score with multiple ignoring parameters combined.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US #10.3 BILLION",
            "The EIFFEL TOWER!!, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The **MEIJI RESTORATION**, in 1989, transformed Japan into a modernized world power.",
        ]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore only special characters using regex
        regex_to_ignore = [r"[^\w\s\d]+"]
        sas_result = evaluation_result._calculate_sas(
            output_key="answers",
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=regex_to_ignore,
        )

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_bi_encoder(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score using a Bi-Encoder model.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluation_result = self.create_evaluation_result(predictions, labels)
        sas_result = evaluation_result._calculate_sas(
            output_key="answers", model="sentence-transformers/all-mpnet-base-v2"
        )

        assert sas_result["sas"] == 1.0
        assert sas_result["scores"] == [1.0, 1.0, 1.0]

    @pytest.mark.integration
    def test_sas_cross_encoder(self):
        """
        Test calculation of Semantic Answer Similarity (SAS) Score using a Cross Encoder model.
        """
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluation_result = self.create_evaluation_result(predictions, labels)
        sas_result = evaluation_result._calculate_sas(
            output_key="answers", model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        assert round(sas_result["sas"], 6) == pytest.approx(0.999967)
        assert sas_result["scores"] == [
            pytest.approx(0.9999765157699585),
            pytest.approx(0.999968409538269),
            pytest.approx(0.9999572038650513),
        ]
