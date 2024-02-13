import pytest

from haystack.components.eval import SASEvaluator
from haystack.utils.device import ComponentDevice


class TestSASEvaluator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        labels = ["label1", "label2", "label3"]
        evaluator = SASEvaluator(labels=labels)

        assert evaluator._labels == labels
        assert evaluator._regexes_to_ignore is None
        assert evaluator._ignore_case is False
        assert evaluator._ignore_punctuation is False
        assert evaluator._ignore_numbers is False
        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device is None
        assert evaluator._token.resolve_value() == "fake-token"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")

        labels = ["label1", "label2", "label3"]
        evaluator = SASEvaluator(labels=labels, device=ComponentDevice.from_str("cuda:0"))

        expected_dict = {
            "type": "haystack.components.eval.sas_evaluator.SASEvaluator",
            "init_parameters": {
                "labels": labels,
                "regexes_to_ignore": None,
                "ignore_case": False,
                "ignore_punctuation": False,
                "ignore_numbers": False,
                "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "batch_size": 32,
                "device": {"type": "single", "device": "cuda:0"},
                "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN"], "strict": False},
            },
        }
        assert evaluator.to_dict() == expected_dict

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        evaluator = SASEvaluator.from_dict(
            {
                "type": "haystack.components.eval.sas_evaluator.SASEvaluator",
                "init_parameters": {
                    "labels": ["label1", "label2", "label3"],
                    "regexes_to_ignore": None,
                    "ignore_case": False,
                    "ignore_punctuation": False,
                    "ignore_numbers": False,
                    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "batch_size": 32,
                    "device": {"type": "single", "device": "cuda:0"},
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN"], "strict": False},
                },
            }
        )

        assert evaluator._labels == ["label1", "label2", "label3"]
        assert evaluator._regexes_to_ignore is None
        assert evaluator._ignore_case is False
        assert evaluator._ignore_punctuation is False
        assert evaluator._ignore_numbers is False
        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device.to_torch_str() == "cuda:0"
        assert evaluator._token.resolve_value() == "fake-token"

    @pytest.mark.integration
    def test_run_with_empty_inputs(self):
        evaluator = SASEvaluator(labels=[])
        result = evaluator.run(predictions=[])
        assert len(result) == 2
        assert result["sas"] == 0.0
        assert result["scores"] == [0.0]

    @pytest.mark.integration
    def test_run_with_different_lengths(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        evaluator = SASEvaluator(labels=labels)

        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(predictions)

    @pytest.mark.integration
    def test_run_with_matching_predictions(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels)
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)

        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_single_prediction(self):
        labels = ["US $2.3 billion"]
        evaluator = SASEvaluator(labels=labels)

        result = evaluator.run(predictions=["A construction budget of US $2.3 billion"])
        assert len(result) == 2
        assert result["sas"] == pytest.approx(0.689089, abs=1e-5)
        assert result["scores"] == pytest.approx([0.689089], abs=1e-5)

    @pytest.mark.integration
    def test_run_with_mismatched_predictions(self):
        labels = [
            "US $2.3 billion",
            "Paris's cultural magnificence is symbolized by the Eiffel Tower",
            "Japan was transformed into a modernized world power after the Meiji Restoration.",
        ]
        evaluator = SASEvaluator(labels=labels)
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(0.8227189)
        assert result["scores"] == pytest.approx([0.689089, 0.870389, 0.908679], abs=1e-5)

    @pytest.mark.integration
    def test_run_with_ignore_case(self):
        labels = [
            "A construction budget of US $2.3 BILLION",
            "The EIFFEL TOWER, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The MEIJI RESTORATION in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, ignore_case=True)
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_ignore_punctuation(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower completed in 1889 symbolizes Paris's cultural magnificence",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power",
        ]
        evaluator = SASEvaluator(labels=labels, ignore_punctuation=True)
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_ignore_numbers(self):
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, ignore_numbers=True)
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_regex_to_ignore(self):
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, regexes_to_ignore=[r"\d+"])
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_multiple_regex_to_ignore(self):
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, regexes_to_ignore=[r"\d+", r"[^\w\s]"])
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_multiple_ignore_parameters(self):
        labels = [
            "A construction budget of US $10.3 billion",
            "The Eiffel Tower, completed in 2005, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1989, transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(
            labels=labels,
            ignore_numbers=True,
            ignore_punctuation=True,
            ignore_case=True,
            regexes_to_ignore=[r"[^\w\s\d]+"],
        )
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration, in 1868, transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_bi_encoder_model(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, model="sentence-transformers/all-mpnet-base-v2")
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(1.0)
        assert result["scores"] == pytest.approx([1.0, 1.0, 1.0])

    @pytest.mark.integration
    def test_run_with_cross_encoder_model(self):
        labels = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        evaluator = SASEvaluator(labels=labels, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(predictions=predictions)
        assert len(result) == 2
        assert result["sas"] == pytest.approx(0.999967, abs=1e-5)
        assert result["scores"] == pytest.approx([0.9999765157699585, 0.999968409538269, 0.9999572038650513], abs=1e-5)
