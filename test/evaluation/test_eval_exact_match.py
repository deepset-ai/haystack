import numpy as np
import pytest

from haystack.evaluation.eval import EvaluationResult


class TestExactMatch:
    @pytest.fixture
    def evaluation_result(self):
        runnable = None
        inputs = []
        outputs = []
        expected_outputs = []
        eval_result = EvaluationResult(runnable, inputs, outputs, expected_outputs)
        return eval_result

    def test_exact_match(self, evaluation_result):
        """
        Test exact match with default parameters
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        em_result = evaluation_result._calculate_em(predictions, labels)

        assert em_result["exact_match"] == 1.0

    def test_exact_match_empty_inputs(self, evaluation_result):
        """
        Test exact match with empty inputs
        """
        predictions = []
        labels = []
        # Expecting 0% exact match for empty inputs
        em_result = evaluation_result._calculate_em(predictions, labels)
        assert em_result["exact_match"] == 0.0

    def test_exact_match_different_data_types(self, evaluation_result):
        """
        Test exact match with different data types (numpy arrays)
        """
        predictions = np.array(["OpenSource", "HaystackAI", "LLMs"])
        labels = np.array(["OpenSource", "HaystackAI", "LLMs"])

        em_result = evaluation_result._calculate_em(predictions, labels)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_single_word(self, evaluation_result):
        """
        Test exact match with single-word inputs
        """
        predictions = ["OpenSource"]
        labels = ["OpenSource"]

        em_result = evaluation_result._calculate_em(predictions, labels)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_negative_case(self, evaluation_result):
        """
        Test exact match with deliberately mismatched predictions and labels
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["Source", "HaystackAI", "LLMs"]
        expected_em = 2 / 3  # Expecting EM to be 2/3 as 2 out of 3 items match
        em_result = evaluation_result._calculate_em(predictions, labels)
        assert em_result["exact_match"] == expected_em

    def test_exact_match_ignore_case(self, evaluation_result):
        """
        Test exact match with ignoring case sensitivity
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["opensource", "HAYSTACKAI", "llMs"]

        # Exact match after case ignoring
        em_result = evaluation_result._calculate_em(predictions, labels, ignore_case=True)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_ignore_punctuation(self, evaluation_result):
        """
        Test exact match with ignoring punctuation
        """
        predictions = ["OpenSource!", "Haystack.AI", "LLMs,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        # Exact match after ignoring punctuation
        em_result = evaluation_result._calculate_em(predictions, labels, ignore_punctuation=True)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_ignore_numbers(self, evaluation_result):
        """
        Test exact match with ignoring numbers
        """
        predictions = ["OpenSource123", "HaystackAI", "LLMs456"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        # Exact match after ignoring numbers
        em_result = evaluation_result._calculate_em(predictions, labels, ignore_numbers=True)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_regex_ignore(self, evaluation_result):
        """
        Test exact match with ignoring specific regex patterns
        """
        predictions = ["Open123Source", "HaystackAI", "LLMs456"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        # Ignore numeric patterns
        regex_to_ignore = [r"\d+"]
        em_result = evaluation_result._calculate_em(predictions, labels, regexes_to_ignore=regex_to_ignore)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_multiple_ignore_regex(self, evaluation_result):
        """
        Test exact match with multiple ignoring parameters
        """
        predictions = ["Open123!Source", "Haystack.AI", "LLMs456,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        # Ignore numeric patterns and punctuation using regex
        regex_to_ignore = [r"\d+", r"\W+"]
        em_result = evaluation_result._calculate_em(predictions, labels, regexes_to_ignore=regex_to_ignore)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_multiple_ignore_combination(self, evaluation_result):
        """
        Test exact match with multiple ignoring parameters combined
        """
        predictions = ["Open%123!$Source", "Haystack.AI##", "^^LLMs456,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        # Ignore only special characters using regex
        regex_to_ignore = [r"[^\w\s\d]+"]
        em_result = evaluation_result._calculate_em(
            predictions,
            labels,
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=regex_to_ignore,
        )
        assert em_result["exact_match"] == 1.0
