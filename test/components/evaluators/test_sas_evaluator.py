# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.utils.device import ComponentDevice


class TestSASEvaluator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        evaluator = SASEvaluator()

        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device is None
        assert evaluator._token.resolve_value() == "fake-token"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")

        evaluator = SASEvaluator(device=ComponentDevice.from_str("cuda:0"))

        expected_dict = {
            "type": "haystack.components.evaluators.sas_evaluator.SASEvaluator",
            "init_parameters": {
                "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "batch_size": 32,
                "device": {"type": "single", "device": "cuda:0"},
                "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
            },
        }
        assert evaluator.to_dict() == expected_dict

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        evaluator = SASEvaluator.from_dict(
            {
                "type": "haystack.components.evaluators.sas_evaluator.SASEvaluator",
                "init_parameters": {
                    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "batch_size": 32,
                    "device": {"type": "single", "device": "cuda:0"},
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                },
            }
        )

        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device.to_torch_str() == "cuda:0"
        assert evaluator._token.resolve_value() == "fake-token"

    def test_run_with_empty_inputs(self):
        evaluator = SASEvaluator()
        result = evaluator.run(ground_truth_answers=[], predicted_answers=[])
        assert len(result) == 2
        assert result["score"] == 0.0
        assert result["individual_scores"] == [0.0]

    def test_run_with_different_lengths(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)

    def test_run_with_none_in_predictions(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            None,
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)

    def test_run_not_warmed_up(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(RuntimeError):
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)

    @pytest.mark.integration
    def test_run_with_matching_predictions(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator.warm_up()
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)

        assert len(result) == 2
        assert result["score"] == pytest.approx(1.0)
        assert result["individual_scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_single_prediction(self):
        evaluator = SASEvaluator()

        ground_truths = ["US $2.3 billion"]
        evaluator.warm_up()
        result = evaluator.run(
            ground_truth_answers=ground_truths, predicted_answers=["A construction budget of US $2.3 billion"]
        )
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.689089, abs=1e-5)
        assert result["individual_scores"] == pytest.approx([0.689089], abs=1e-5)

    @pytest.mark.integration
    def test_run_with_mismatched_predictions(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "US $2.3 billion",
            "Paris's cultural magnificence is symbolized by the Eiffel Tower",
            "Japan was transformed into a modernized world power after the Meiji Restoration.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator.warm_up()
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.8227189)
        assert result["individual_scores"] == pytest.approx([0.689089, 0.870389, 0.908679], abs=1e-5)

    @pytest.mark.integration
    def test_run_with_bi_encoder_model(self):
        evaluator = SASEvaluator(model="sentence-transformers/all-mpnet-base-v2")
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator.warm_up()
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(1.0)
        assert result["individual_scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_cross_encoder_model(self):
        evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator.warm_up()
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.999967, abs=1e-5)
        assert result["individual_scores"] == pytest.approx(
            [0.9999765157699585, 0.999968409538269, 0.9999572038650513], abs=1e-5
        )
