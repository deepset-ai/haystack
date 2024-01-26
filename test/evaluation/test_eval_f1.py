import pytest

from haystack import Pipeline
from haystack.dataclasses import GeneratedAnswer
from haystack.evaluation.eval import EvaluationResult


class TestF1:
    def create_evaluation_result(self, predictions, labels):
        """
        Creates an evaluation result of a RAG pipeline using the list of predictions and labels for testing the f1.
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

    def test_f1_empty_inputs(self):
        """
        Test f1 with empty inputs
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
        # Expecting 0% f1 for empty inputs
        f1_result = evaluation_result._calculate_f1(output_key="answers")

        assert f1_result["f1"] == 0.0

    def test_calculate_f1_with_different_lengths(self):
        """
        Test f1 with default parameters
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["OpenSource", "HaystackAI"]
        evaluation_result = self.create_evaluation_result(predictions, labels)

        with pytest.raises(ValueError, match="The number of predictions and labels must be the same."):
            evaluation_result._calculate_f1(output_key="answers")

    def test_f1_same_inputs(self):
        """
        Test f1 with default parameters
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluation_result = self.create_evaluation_result(predictions, labels)
        f1_result = evaluation_result._calculate_f1(output_key="answers")

        assert f1_result["f1"] == 1.0

    def test_f1_single_word(self):
        """
        Test f1 with single-word inputs
        """
        predictions = ["Open Source"]
        labels = ["Source"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        f1_result = evaluation_result._calculate_f1(output_key="answers")

        assert f1_result["f1"] == pytest.approx(2 / 3)

    def test_f1_negative_case(self):
        """
        Test f1 with deliberately mismatched predictions and labels
        """
        predictions = ["Open Source", "HaystackAI"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        f1_result = evaluation_result._calculate_f1(output_key="answers")

        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_ignore_case(self):
        """
        Test f1 with ignoring case sensitivity
        """
        predictions = ["Open Source", "HaystackAI"]
        labels = ["source", "HAYSTACKAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # F1 after case ignoring
        f1_result = evaluation_result._calculate_f1(output_key="answers", ignore_case=True)

        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_ignore_punctuation(self):
        """
        Test f1 with ignoring punctuation
        """
        predictions = ["Open Source!", "Haystack.AI"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # F1 after ignoring punctuation
        f1_result = evaluation_result._calculate_f1(output_key="answers", ignore_punctuation=True)

        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_ignore_numbers(self):
        """
        Test f1 with ignoring numbers
        """
        predictions = ["Open Source123", "HaystackAI"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # F1 after ignoring numbers
        f1_result = evaluation_result._calculate_f1(output_key="answers", ignore_numbers=True)
        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_regex_ignore(self):
        """
        Test f1 with ignoring specific regex patterns
        """
        predictions = ["Open123 Source", "HaystackAI"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns
        regex_to_ignore = [r"\d+"]
        f1_result = evaluation_result._calculate_f1(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_multiple_ignore_regex(self):
        """
        Test f1 with multiple ignoring parameters
        """
        predictions = ["Open123! Source", "Haystack.AI"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns and punctuation excluding whitespaces
        regex_to_ignore = [r"\d+", r"[^\w\s]"]
        f1_result = evaluation_result._calculate_f1(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert f1_result["f1"] == pytest.approx(5 / 6)

    def test_f1_multiple_ignore_combination(self):
        """
        Test f1 with multiple ignoring parameters combined
        """
        predictions = ["Open%123. !$Source", "Haystack.AI##"]
        labels = ["Source", "HaystackAI"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore only special characters using regex
        regex_to_ignore = [r"[^\w\s\d]+"]
        f1_result = evaluation_result._calculate_f1(
            output_key="answers",
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=regex_to_ignore,
        )

        assert f1_result["f1"] == pytest.approx(5 / 6)
