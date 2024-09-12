# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

import math

import pytest

from haystack import Pipeline
from haystack.components.evaluators import ContextRelevanceEvaluator
from haystack.utils.auth import Secret


class TestContextRelevanceEvaluator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.instructions == (
            "Please extract only sentences from the provided context which are absolutely relevant and "
            "required to answer the following question. If no relevant sentences are found, or if you "
            "believe the question cannot be answered from the given context, return an empty list, example: []"
        )
        assert component.inputs == [("questions", List[str]), ("contexts", List[List[str]])]
        assert component.outputs == ["relevant_statements"]
        assert component.examples == [
            {
                "inputs": {
                    "questions": "What is the capital of Germany?",
                    "contexts": ["Berlin is the capital of Germany. Berlin and was founded in 1244."],
                },
                "outputs": {"relevant_statements": ["Berlin is the capital of Germany."]},
            },
            {
                "inputs": {
                    "questions": "What is the capital of France?",
                    "contexts": [
                        "Berlin is the capital of Germany and was founded in 1244.",
                        "Europe is a continent with 44 countries.",
                        "Madrid is the capital of Spain.",
                    ],
                },
                "outputs": {"relevant_statements": []},
            },
            {
                "inputs": {"questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."]},
                "outputs": {"relevant_statements": ["Rome is the capital of Italy."]},
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

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = ContextRelevanceEvaluator(
            api="openai",
            api_key=Secret.from_env_var("ENV_VAR"),
            examples=[{"inputs": {"questions": "What is football?"}, "outputs": {"score": 0}}],
            raise_on_failure=False,
            progress_bar=False,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.evaluators.context_relevance.ContextRelevanceEvaluator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "api": "openai",
                "api_params": {"generation_kwargs": {"response_format": {"type": "json_object"}, "seed": 42}},
                "examples": [{"inputs": {"questions": "What is football?"}, "outputs": {"score": 0}}],
                "progress_bar": False,
                "raise_on_failure": False,
            },
        }

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

        pipeline = Pipeline()
        pipeline.add_component("evaluator", component)
        assert pipeline.loads(pipeline.dumps())

    def test_run_calculates_mean_score(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()

        def generator_run(self, *args, **kwargs):
            if "Football" in kwargs["prompt"]:
                return {"replies": ['{"relevant_statements": ["a", "b"], "score": 1}']}
            else:
                return {"replies": ['{"relevant_statements": [], "score": 0}']}

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
                "Python is design philosophy emphasizes code readability, and its language constructs aim to help "
                "programmers write clear, logical code for both small and large-scale software projects."
            ],
        ]
        results = component.run(questions=questions, contexts=contexts)

        print(results)

        assert results == {
            "results": [{"score": 1, "relevant_statements": ["a", "b"]}, {"score": 0, "relevant_statements": []}],
            "score": 0.5,
            "meta": None,
            "individual_scores": [1, 0],
        }

    def test_run_no_statements_extracted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()

        def generator_run(self, *args, **kwargs):
            if "Football" in kwargs["prompt"]:
                return {"replies": ['{"relevant_statements": ["a", "b"], "score": 1}']}
            else:
                return {"replies": ['{"relevant_statements": [], "score": 0}']}

        monkeypatch.setattr("haystack.components.generators.openai.OpenAIGenerator.run", generator_run)

        questions = ["Which is the most popular global sport?", "Who created the Python language?"]
        contexts = [
            [
                "The popularity of sports can be measured in various ways, including TV viewership, social media "
                "presence, number of participants, and economic impact. Football is undoubtedly the world's most "
                "popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and "
                "Messi, drawing a followership of more than 4 billion people."
            ],
            [],
        ]
        results = component.run(questions=questions, contexts=contexts)
        assert results == {
            "results": [{"score": 1, "relevant_statements": ["a", "b"]}, {"score": 0, "relevant_statements": []}],
            "score": 0.5,
            "meta": None,
            "individual_scores": [1, 0],
        }

    def test_run_missing_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator()
        with pytest.raises(ValueError, match="LLM evaluator expected input parameter"):
            component.run()

    def test_run_returns_nan_raise_on_failure_false(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = ContextRelevanceEvaluator(raise_on_failure=False)

        def generator_run(self, *args, **kwargs):
            if "Python" in kwargs["prompt"]:
                raise Exception("OpenAI API request failed.")
            else:
                return {"replies": ['{"relevant_statements": ["c", "d"], "score": 1}']}

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

        assert math.isnan(results["score"])
        assert results["results"][0] == {"relevant_statements": ["c", "d"], "score": 1}
        assert results["results"][1]["relevant_statements"] == []
        assert math.isnan(results["results"][1]["score"])

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

        required_fields = {"results"}
        assert all(field in result for field in required_fields)
        nested_required_fields = {"score", "relevant_statements"}
        assert all(field in result["results"][0] for field in nested_required_fields)

        assert "meta" in result
        assert "prompt_tokens" in result["meta"][0]["usage"]
        assert "completion_tokens" in result["meta"][0]["usage"]
        assert "total_tokens" in result["meta"][0]["usage"]
