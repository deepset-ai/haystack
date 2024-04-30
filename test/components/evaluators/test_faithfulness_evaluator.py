import os
from typing import List

import pytest

from haystack.components.evaluators import FaithfulnessEvaluator
from haystack.utils.auth import Secret


class TestFaithfulnessEvaluator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = FaithfulnessEvaluator()
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.instructions == (
            "Your task is to judge the faithfulness or groundedness of statements based "
            "on context information. First, please extract statements from a provided predicted "
            "answer to a question. Second, calculate a faithfulness score for each "
            "statement made in the predicted answer. The score is 1 if the statement can be "
            "inferred from the provided context or 0 if it cannot be inferred."
        )
        assert component.inputs == [
            ("questions", List[str]),
            ("contexts", List[List[str]]),
            ("predicted_answers", List[str]),
        ]
        assert component.outputs == ["statements", "statement_scores"]
        assert component.examples == [
            {
                "inputs": {
                    "questions": "What is the capital of Germany and when was it founded?",
                    "contexts": ["Berlin is the capital of Germany and was founded in 1244."],
                    "predicted_answers": "The capital of Germany, Berlin, was founded in the 13th century.",
                },
                "outputs": {
                    "statements": ["Berlin is the capital of Germany.", "Berlin was founded in 1244."],
                    "statement_scores": [1, 1],
                },
            },
            {
                "inputs": {
                    "questions": "What is the capital of France?",
                    "contexts": ["Berlin is the capital of Germany."],
                    "predicted_answers": "Paris",
                },
                "outputs": {"statements": ["Paris is the capital of France."], "statement_scores": [0]},
            },
            {
                "inputs": {
                    "questions": "What is the capital of Italy?",
                    "contexts": ["Rome is the capital of Italy."],
                    "predicted_answers": "Rome is the capital of Italy with more than 4 million inhabitants.",
                },
                "outputs": {
                    "statements": ["Rome is the capital of Italy.", "Rome has more than 4 million inhabitants."],
                    "statement_scores": [1, 0],
                },
            },
        ]

    def test_init_fail_wo_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            FaithfulnessEvaluator()

    def test_init_with_parameters(self):
        component = FaithfulnessEvaluator(
            api_key=Secret.from_token("test-api-key"),
            api="openai",
            examples=[
                {
                    "inputs": {"predicted_answers": "Damn, this is straight outta hell!!!"},
                    "outputs": {"custom_score": 1},
                },
                {
                    "inputs": {"predicted_answers": "Football is the most popular sport."},
                    "outputs": {"custom_score": 0},
                },
            ],
        )
        assert component.generator.client.api_key == "test-api-key"
        assert component.api == "openai"
        assert component.examples == [
            {"inputs": {"predicted_answers": "Damn, this is straight outta hell!!!"}, "outputs": {"custom_score": 1}},
            {"inputs": {"predicted_answers": "Football is the most popular sport."}, "outputs": {"custom_score": 0}},
        ]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        data = {
            "type": "haystack.components.evaluators.faithfulness.FaithfulnessEvaluator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api": "openai",
                "examples": [
                    {"inputs": {"predicted_answers": "Football is the most popular sport."}, "outputs": {"score": 0}}
                ],
            },
        }
        component = FaithfulnessEvaluator.from_dict(data)
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.examples == [
            {"inputs": {"predicted_answers": "Football is the most popular sport."}, "outputs": {"score": 0}}
        ]

    def test_run_calculates_mean_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = FaithfulnessEvaluator()

        def generator_run(self, *args, **kwargs):
            if "Football" in kwargs["prompt"]:
                return {"replies": ['{"statements": ["a", "b"], "statement_scores": [1, 0]}']}
            else:
                return {"replies": ['{"statements": ["c", "d"], "statement_scores": [1, 1]}']}

        monkeypatch.setattr("haystack.components.generators.openai.OpenAIGenerator.run", generator_run)

        questions = ["Which is the most popular global sport?", "Who created the Python language?"]
        contexts = [
            [
                "The popularity of sports can be measured in various ways, including TV viewership, social media "
                "presence, number of participants, and economic impact. Football is undoubtedly the world's most "
                "popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and "
                "Messi, drawing a followership of more than 4 billion people."
            ],
            [
                "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming "
                "language. Its design philosophy emphasizes code readability, and its language constructs aim to help "
                "programmers write clear, logical code for both small and large-scale software projects."
            ],
        ]
        predicted_answers = [
            "Football is the most popular sport with around 4 billion followers worldwide.",
            "Python is a high-level general-purpose programming language that was created by George Lucas.",
        ]
        results = component.run(questions=questions, contexts=contexts, predicted_answers=predicted_answers)
        assert results == {
            "individual_scores": [0.5, 1],
            "results": [
                {"score": 0.5, "statement_scores": [1, 0], "statements": ["a", "b"]},
                {"score": 1, "statement_scores": [1, 1], "statements": ["c", "d"]},
            ],
            "score": 0.75,
        }

    def test_run_missing_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = FaithfulnessEvaluator()
        with pytest.raises(TypeError, match="missing 3 required positional arguments"):
            component.run()

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        questions = ["What is Python and who created it?"]
        contexts = [["Python is a programming language created by Guido van Rossum."]]
        predicted_answers = ["Python is a programming language created by George Lucas."]
        evaluator = FaithfulnessEvaluator()
        result = evaluator.run(questions=questions, contexts=contexts, predicted_answers=predicted_answers)

        required_fields = {"individual_scores", "results", "score"}
        assert all(field in result for field in required_fields)
        nested_required_fields = {"score", "statement_scores", "statements"}
        assert all(field in result["results"][0] for field in nested_required_fields)
