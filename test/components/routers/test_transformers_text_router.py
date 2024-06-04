# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from haystack.components.routers.transformers_text_router import TransformersTextRouter
from haystack.utils import ComponentDevice, Secret


class TestTransformersZeroShotTextRouter:
    def test_to_dict(self):
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router_dict = router.to_dict()
        assert router_dict == {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "text-classification",
                },
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.transformers_text_router.TransformersTextRouter",
            "init_parameters": {
                "labels": ["query", "passage"],
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
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
        assert component.token == Secret.from_dict({"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"})
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "text-classification",
            "token": None,
        }

    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_warm_up(self, hf_pipeline_mock):
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.warm_up()
        assert router.pipeline is not None

    def test_run_error(self):
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        with pytest.raises(RuntimeError):
            router.run(text="test")

    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_run_not_str_error(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = " "
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router.warm_up()
        with pytest.raises(TypeError):
            router.run(text=["wrong_input"])

    @patch("haystack.components.routers.transformers_text_router.pipeline")
    def test_run_unit(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = [
            {"sequence": "What is the color of the sky?", "labels": ["query", "passage"], "scores": [0.9, 0.1]}
        ]
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
        assert router.pipeline is not None
        assert out == {"en": "What is the color of the sky?"}
