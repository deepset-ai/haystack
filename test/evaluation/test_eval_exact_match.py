from haystack import Pipeline
from haystack.dataclasses import GeneratedAnswer
from haystack.evaluation.eval import EvaluationResult


class TestExactMatch:
    def create_evaluation_result(self, predictions, labels):
        """
        Creates an evaluation result of a RAG pipeline using the list of predictions and labels for testing the exact match.
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

    def test_exact_match_empty_inputs(self):
        """
        Test exact match with empty inputs
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
        # Expecting 0% exact match for empty inputs
        em_result = evaluation_result._calculate_em(output_key="answers")

        assert em_result["exact_match"] == 0.0

    def test_exact_match_same_inputs(self):
        """
        Test exact match with default parameters
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]
        evaluation_result = self.create_evaluation_result(predictions, labels)
        em_result = evaluation_result._calculate_em(output_key="answers")

        assert em_result["exact_match"] == 1.0

    def test_exact_match_single_word(self):
        """
        Test exact match with single-word inputs
        """
        predictions = ["OpenSource"]
        labels = ["OpenSource"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        em_result = evaluation_result._calculate_em(output_key="answers")

        assert em_result["exact_match"] == 1.0

    def test_exact_match_negative_case(self):
        """
        Test exact match with deliberately mismatched predictions and labels
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["Source", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Expecting EM to be 2/3 as 2 out of 3 items match
        expected_em = 2 / 3
        em_result = evaluation_result._calculate_em(output_key="answers")

        assert em_result["exact_match"] == expected_em

    def test_exact_match_ignore_case(self):
        """
        Test exact match with ignoring case sensitivity
        """
        predictions = ["OpenSource", "HaystackAI", "LLMs"]
        labels = ["opensource", "HAYSTACKAI", "llMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Exact match after case ignoring
        em_result = evaluation_result._calculate_em(output_key="answers", ignore_case=True)

        assert em_result["exact_match"] == 1.0

    def test_exact_match_ignore_punctuation(self):
        """
        Test exact match with ignoring punctuation
        """
        predictions = ["OpenSource!", "Haystack.AI", "LLMs,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Exact match after ignoring punctuation
        em_result = evaluation_result._calculate_em(output_key="answers", ignore_punctuation=True)

        assert em_result["exact_match"] == 1.0

    def test_exact_match_ignore_numbers(self):
        """
        Test exact match with ignoring numbers
        """
        predictions = ["OpenSource123", "HaystackAI", "LLMs456"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Exact match after ignoring numbers
        em_result = evaluation_result._calculate_em(output_key="answers", ignore_numbers=True)
        assert em_result["exact_match"] == 1.0

    def test_exact_match_regex_ignore(self):
        """
        Test exact match with ignoring specific regex patterns
        """
        predictions = ["Open123Source", "HaystackAI", "LLMs456"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns
        regex_to_ignore = [r"\d+"]
        em_result = evaluation_result._calculate_em(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert em_result["exact_match"] == 1.0

    def test_exact_match_multiple_ignore_regex(self):
        """
        Test exact match with multiple ignoring parameters
        """
        predictions = ["Open123!Source", "Haystack.AI", "LLMs456,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore numeric patterns and punctuation using regex
        regex_to_ignore = [r"\d+", r"\W+"]
        em_result = evaluation_result._calculate_em(output_key="answers", regexes_to_ignore=regex_to_ignore)

        assert em_result["exact_match"] == 1.0

    def test_exact_match_multiple_ignore_combination(self):
        """
        Test exact match with multiple ignoring parameters combined
        """
        predictions = ["Open%123!$Source", "Haystack.AI##", "^^LLMs456,"]
        labels = ["OpenSource", "HaystackAI", "LLMs"]

        evaluation_result = self.create_evaluation_result(predictions, labels)
        # Ignore only special characters using regex
        regex_to_ignore = [r"[^\w\s\d]+"]
        em_result = evaluation_result._calculate_em(
            output_key="answers",
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=regex_to_ignore,
        )

        assert em_result["exact_match"] == 1.0
