# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest
from sentence_transformers import CrossEncoder
from torch.nn import Identity, Sigmoid

from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.utils import expit
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
        assert evaluator._device is not None
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
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("activation_fn", "model_scores", "expected_scores"),
        [
            # Raw logits are normalized with a sigmoid
            (Identity(), {"Lyon": -9.4, "Paris": 4.9}, {"Lyon": expit(-9.4), "Paris": expit(4.9)}),
            # Scores that the model already normalized are returned as they are
            (Sigmoid(), {"Lyon": 0.2, "Paris": 0.9}, {"Lyon": 0.2, "Paris": 0.9}),
        ],
    )
    def test_run_cross_encoder_score_does_not_depend_on_batch(
        self, activation_fn, model_scores, expected_scores, monkeypatch
    ):
        # Use the mocked model below instead of loading a real one
        monkeypatch.setattr(SASEvaluator, "warm_up", lambda self: None)
        model = MagicMock(spec=CrossEncoder)
        model.num_labels = 1
        model.activation_fn = activation_fn
        model.predict.side_effect = lambda pairs, **kwargs: np.array([model_scores[pred] for pred, _ in pairs])
        evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        evaluator._similarity_model = model

        wrong_only = evaluator.run(ground_truth_answers=["Paris"], predicted_answers=["Lyon"])
        mixed = evaluator.run(ground_truth_answers=["Paris", "Paris"], predicted_answers=["Lyon", "Paris"])

        assert wrong_only["individual_scores"] == pytest.approx([expected_scores["Lyon"]])
        assert mixed["individual_scores"] == pytest.approx([expected_scores["Lyon"], expected_scores["Paris"]])

    def test_run_with_multi_label_cross_encoder_raises(self, monkeypatch):
        # The mocked model has a length of 0, so run would try to load the real model without this
        monkeypatch.setattr(SASEvaluator, "warm_up", lambda self: None)
        evaluator = SASEvaluator(model="cross-encoder/nli-deberta-v3-xsmall")
        model = MagicMock(spec=CrossEncoder)
        model.num_labels = 3
        model.predict.return_value = np.array([[2.0, -1.0, 0.0]])
        evaluator._similarity_model = model

        with pytest.raises(ValueError, match="3"):
            evaluator.run(ground_truth_answers=["Paris"], predicted_answers=["Lyon"])
        model.predict.assert_not_called()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_with_bi_encoder_model(self, del_hf_env_vars):
        evaluator = SASEvaluator("sentence-transformers-testing/stsb-bert-tiny-safetensors")
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
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.912335)
        assert result["individual_scores"] == pytest.approx([0.855047, 0.907907, 0.974050], abs=1e-5)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_with_cross_encoder_model(self, del_hf_env_vars):
        evaluator = SASEvaluator(model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
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
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.938108, abs=1e-5)
        assert result["individual_scores"] == pytest.approx([0.930112, 0.9431504, 0.9410622], abs=1e-5)
