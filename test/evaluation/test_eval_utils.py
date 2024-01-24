from haystack.dataclasses import GeneratedAnswer
from haystack.evaluation.eval_utils import get_answers_from_output, preprocess_text


class TestEvalUtils:
    def test_extract_answers_from_pipeline_output(self):
        """
        Test that the function correctly extracts answers from the output of a pipeline.
        """
        outputs = [
            {
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={})]
                }
            },
            {
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]
                }
            },
            {
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Giorgio", query="Who lives in Rome?", documents=[], meta={})]
                }
            },
        ]

        runnable_type = "pipeline"
        output_key = "answers"
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_extract_answers_from_component_output(self):
        """
        Test that the function correctly extracts answers from the output of a component.
        """
        outputs = [
            {"answers": [GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={})]},
            {"answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]},
            {"answers": [GeneratedAnswer(data="Giorgio", query="Who lives in Rome?", documents=[], meta={})]},
        ]
        runnable_type = "component"
        output_key = "answers"
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_ignore_other_output_keys(self):
        """
        Test that the function only extracts answers and ignores other output keys.
        """
        outputs = [
            {
                "llm": {"replies": ["llm_reply_1"]},
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={})]
                },
            },
            {
                "llm": {"replies": ["llm_reply_2"]},
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]
                },
            },
            {
                "llm": {"replies": ["llm_reply_3"]},
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Giorgio", query="Who lives in Rome?", documents=[], meta={})]
                },
            },
        ]

        runnable_type = "pipeline"
        output_key = "answers"
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_handle_empty_outputs(self):
        """
        Test that the function correctly handles empty outputs.
        """
        outputs = []
        runnable_type = "pipeline"
        output_key = "answers"
        expected_answers = []

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_handle_missing_keys(self):
        """
        Test that the function correctly handles outputs with missing keys.
        """
        outputs = [
            {
                "llm": {"replies": ["llm_reply_1"]},
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={})]
                },
            },
            {
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]
                }
            },
        ]

        runnable_type = "pipeline"
        output_key = "answers"
        expected_answers = ["Jean", "Mark"]

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_handle_missing_values(self):
        """
        Test that the function correctly handles outputs with missing values.
        """
        outputs = [
            {"answer_builder": {"answers": []}},
            {
                "answer_builder": {
                    "answers": [GeneratedAnswer(data="Mark", query="Who lives in Berlin?", documents=[], meta={})]
                }
            },
        ]
        runnable_type = "pipeline"
        output_key = "answers"
        expected_answers = ["Mark"]

        assert get_answers_from_output(outputs, output_key, runnable_type) == expected_answers

    def test_preprocess_text_default_parameters(self):
        """
        Test preprocess_text with default parameters.
        There should be no changes to the input text.
        """
        texts = ["Test, Output-1!", "Test, Output-2!"]
        expected_output = ["Test, Output-1!", "Test, Output-2!"]
        actual_output = preprocess_text(texts)

        assert actual_output == expected_output

    def test_preprocess_text_ignore_case(self):
        """
        Test preprocess_text with ignore_case=True.

        """
        texts = ["Test, Output-1!"]
        expected_output = ["test, output-1!"]

        actual_output = preprocess_text(texts, ignore_case=True)

        assert actual_output == expected_output

    def test_preprocess_text_ignore_punctuation(self):
        """
        Test preprocess_text with ignore_punctuation=True.
        """
        texts = ["Test, Output-1!"]
        expected_output = ["Test Output1"]

        actual_output = preprocess_text(texts, ignore_punctuation=True)

        assert actual_output == expected_output

    # Preprocess text with ignore_numbers=True.
    def test_preprocess_text_ignore_numbers(self):
        """
        Test preprocess_text with ignore_numbers=True. It should be able to remove numbers from the input.
        """
        texts = ["Test, Output-1!"]
        expected_output = ["Test, Output-!"]

        actual_output = preprocess_text(texts, ignore_numbers=True)

        assert actual_output == expected_output

    def test_preprocess_text_regexes_to_ignore(self):
        """
        Test preprocess_text with a list of regex patterns to ignore.
        """
        texts = ["Test, Output-1!"]
        expected_output = ["Test Output"]

        # Use regex patterns to remove digits and non-alphanumeric characters
        actual_output = preprocess_text(texts, regexes_to_ignore=[r"\d", r"[^\w\s]"])

        assert actual_output == expected_output

    def test_preprocess_text_empty_list(self):
        """
        Test preprocess_text with empty list of texts.
        """
        texts = []
        expected_output = []

        actual_output = preprocess_text(texts)

        assert actual_output == expected_output

    def test_preprocess_text_all_ignore_parameters(self):
        """
        Test preprocess_text with all ignore parameters set to True.
        """
        texts = ["Test, Output-1!"]
        expected_output = ["test output"]

        actual_output = preprocess_text(texts, ignore_case=True, ignore_punctuation=True, ignore_numbers=True)

        assert actual_output == expected_output

    def test_preprocess_text_regexes_to_ignore_empty_string(self):
        """
        Test preprocess_text with regexes_to_ignore=[""].
        """
        texts = ["Test, Output-1!"]
        expected_output = ["Test, Output-1!"]

        actual_output = preprocess_text(texts, regexes_to_ignore=[""])

        assert actual_output == expected_output

    # Preprocess text with regexes_to_ignore=[".*"].
    def test_preprocess_text_regexes_to_ignore_dot_star(self):
        """
        Test preprocess_text with regexes_to_ignore=[".*"].
        """
        texts = ["Test, Output-1!"]
        expected_output = [""]

        actual_output = preprocess_text(texts, regexes_to_ignore=[".*"])

        assert actual_output == expected_output

    def test_preprocess_text_regexes_to_ignore_same_substring(self):
        """
        Test preprocess_text with regexes_to_ignore where all the regex patterns match the same substring.
        """
        texts = ["Test, Output-1!"]
        expected_output = ["Test, Output-!"]

        actual_output = preprocess_text(texts, regexes_to_ignore=[r"\d", r"\d"])

        assert actual_output == expected_output
