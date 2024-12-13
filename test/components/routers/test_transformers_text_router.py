# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch, MagicMock

import pytest

from haystack.components.routers.transformers_text_router import TransformersTextRouter
from haystack.utils import ComponentDevice, Secret


class TestTransformersTextRouter:
    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_to_dict(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router_dict = router.to_dict()
        assert router_dict == {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "labels": ["en", "de"],
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "text-classification",
                },
            },
        }

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_to_dict_with_cpu_device(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(
            model="papluca/xlm-roberta-base-language-detection", device=ComponentDevice.from_str("cpu")
        )
        router_dict = router.to_dict()
        assert router_dict == {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "labels": ["en", "de"],
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.from_str("cpu").to_hf(),
                    "task": "text-classification",
                },
            },
        }

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_from_dict(self, mock_auto_config_from_pretrained, monkeypatch):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "text-classification",
            "token": None,
        }

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_from_dict_no_default_parameters(self, mock_auto_config_from_pretrained, monkeypatch):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {"model": "papluca/xlm-roberta-base-language-detection"},
        }
        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "text-classification",
            "token": None,
        }

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_from_dict_with_cpu_device(self, mock_auto_config_from_pretrained, monkeypatch):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.from_str("cpu").to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.from_str("cpu").to_hf(),
            "task": "text-classification",
            "token": None,
        }

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_warm_up(self, hf_pipeline_mock, mock_auto_config_from_pretrained):
        hf_pipeline_mock.return_value = MagicMock(model=MagicMock(config=MagicMock(label2id={"en": 0, "de": 1})))
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.warm_up()
        assert router.pipeline is not None

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    def test_run_fails_without_warm_up(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        with pytest.raises(RuntimeError):
            router.run(text="test")

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_run_fails_with_non_string_input(self, hf_pipeline_mock, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        hf_pipeline_mock.return_value = MagicMock(model=MagicMock(config=MagicMock(label2id={"en": 0, "de": 1})))
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.warm_up()
        with pytest.raises(TypeError):
            router.run(text=["wrong_input"])

    @patch("haystack.components.routers.transformers_text_router.AutoConfig.from_pretrained")
    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_run_unit(self, hf_pipeline_mock, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        hf_pipeline_mock.return_value = [{"label": "en", "score": 0.9}]
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.pipeline = hf_pipeline_mock
        out = router.run("What is the color of the sky?")
        assert router.pipeline is not None
        assert out == {"en": "What is the color of the sky?"}

    @pytest.mark.integration
    def test_run(self):
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.warm_up()
        out = router.run("What is the color of the sky?")
        assert set(router.labels) == {
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "it",
            "ja",
            "nl",
            "pl",
            "pt",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        }
        assert router.pipeline is not None
        assert out == {"en": "What is the color of the sky?"}

    @pytest.mark.integration
    def test_wrong_labels(self):
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection", labels=["en", "de"])
        with pytest.raises(ValueError):
            router.warm_up()
