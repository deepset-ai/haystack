from haystack.dataclasses import GeneratedAnswer
from haystack.evaluation.eval_utils import get_answers_from_output


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
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, runnable_type) == expected_answers

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
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, runnable_type) == expected_answers

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
        expected_answers = ["Jean", "Mark", "Giorgio"]

        assert get_answers_from_output(outputs, runnable_type) == expected_answers

    def test_handle_empty_outputs(self):
        """
        Test that the function correctly handles empty outputs.
        """
        outputs = []
        runnable_type = "pipeline"
        expected_answers = []

        assert get_answers_from_output(outputs, runnable_type) == expected_answers

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
        expected_answers = ["Jean", "Mark"]

        assert get_answers_from_output(outputs, runnable_type) == expected_answers

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
        expected_answers = ["Mark"]

        assert get_answers_from_output(outputs, runnable_type) == expected_answers
