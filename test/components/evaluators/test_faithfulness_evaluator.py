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
        assert component.instructions == "test-instruction"
        assert component.inputs == [("responses", List[str])]
        assert component.outputs == ["score"]
        assert component.examples == [
            {"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"score": 0}}
        ]

    def test_init_fail_wo_openai_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            FaithfulnessEvaluator()

    def test_init_with_parameters(self):
        component = FaithfulnessEvaluator(
            instructions="test-instruction",
            api_key=Secret.from_token("test-api-key"),
            inputs=[("responses", List[str])],
            outputs=["custom_score"],
            api="openai",
            examples=[
                {"inputs": {"responses": "Damn, this is straight outta hell!!!"}, "outputs": {"custom_score": 1}},
                {"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"custom_score": 0}},
            ],
        )
        assert component.generator.client.api_key == "test-api-key"
        assert component.api == "openai"
        assert component.examples == [
            {"inputs": {"responses": "Damn, this is straight outta hell!!!"}, "outputs": {"custom_score": 1}},
            {"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"custom_score": 0}},
        ]
        assert component.instructions == "test-instruction"
        assert component.inputs == [("responses", List[str])]
        assert component.outputs == ["custom_score"]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        data = {
            "type": "haystack.components.evaluators.llm_evaluator.FaithfulnessEvaluator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api": "openai",
                "instructions": "test-instruction",
                "inputs": [("responses", List[str])],
                "outputs": ["score"],
                "examples": [{"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"score": 0}}],
            },
        }
        component = FaithfulnessEvaluator.from_dict(data)
        assert component.api == "openai"
        assert component.generator.client.api_key == "test-api-key"
        assert component.instructions == "test-instruction"
        assert component.inputs == [("responses", List[str])]
        assert component.outputs == ["score"]
        assert component.examples == [
            {"inputs": {"responses": "Football is the most popular sport."}, "outputs": {"score": 0}}
        ]
