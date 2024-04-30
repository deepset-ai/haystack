import os
from typing import List

import pytest

from haystack.components.evaluators import ContextRelevanceEvaluator
from haystack.utils.auth import Secret


class TestContextRelevanceEvaluator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.instructions == (
            "Your task is to judge how relevant the provided context is for answering a question. "
            "First, please extract statements from the provided context. "
            "Second, calculate a relevance score for each statement in the context. "
            "The score is 1 if the statement is relevant to answer the question or 0 if it is not relevant."
        )
        assert component.inputs == [("questions", List[str]), ("contexts", List[List[str]])]
        assert component.outputs == ["statements", "statement_scores"]
        assert component.examples == [
            {
                "inputs": {
                    "questions": "What is the capital of Germany?",
                    "contexts": ["Berlin is the capital of Germany and was founded in 1244."],
                },
                "outputs": {
                    "statements": ["Berlin is the capital of Germany.", "Berlin was founded in 1244."],
                    "statement_scores": [1, 0],
                },
            },
            {
                "inputs": {
                    "questions": "What is the capital of France?",
                    "contexts": ["Berlin is the capital of Germany."],
                },
                "outputs": {"statements": ["Berlin is the capital of Germany."], "statement_scores": [0]},
            },
            {
                "inputs": {"questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."]},
                "outputs": {"statements": ["Rome is the capital of Italy."], "statement_scores": [1]},
            },
        ]

    def test_init_fail_wo_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            ContextRelevanceEvaluator()

    def test_init_with_parameters(self):
        component = ContextRelevanceEvaluator(
            api_key=Secret.from_token("test-api-key"),
            api="openai",
            examples=[
                {"inputs": {"questions": "Damn, this is straight outta hell!!!"}, "outputs": {"custom_score": 1}},
                {"inputs": {"questions": "Football is the most popular sport."}, "outputs": {"custom_score": 0}},
            ],
        )
        assert component.generator.client.api_key == "test-api-key"
        assert component.api == "openai"
        assert component.examples == [
            {"inputs": {"questions": "Damn, this is straight outta hell!!!"}, "outputs": {"custom_score": 1}},
            {"inputs": {"questions": "Football is the most popular sport."}, "outputs": {"custom_score": 0}},
        ]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        data = {
            "type": "haystack.components.evaluators.context_relevance.ContextRelevanceEvaluator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api": "openai",
                "examples": [{"inputs": {"questions": "What is football?"}, "outputs": {"score": 0}}],
            },
        }
        component = ContextRelevanceEvaluator.from_dict(data)
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.examples == [{"inputs": {"questions": "What is football?"}, "outputs": {"score": 0}}]

    def test_run_calculates_mean_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()

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
        results = component.run(questions=questions, contexts=contexts)
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
        component = ContextRelevanceEvaluator()
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            component.run()

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        questions = ["Who created the Python language?"]
        contexts = [["Python, created by Guido van Rossum, is a high-level general-purpose programming language."]]

        evaluator = ContextRelevanceEvaluator()
        result = evaluator.run(questions=questions, contexts=contexts)

        required_fields = {"individual_scores", "results", "score"}
        assert all(field in result for field in required_fields)
        nested_required_fields = {"score", "statement_scores", "statements"}
        assert all(field in result["results"][0] for field in nested_required_fields)
