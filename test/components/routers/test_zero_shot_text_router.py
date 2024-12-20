# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from haystack.components.routers.zero_shot_text_router import TransformersZeroShotTextRouter
from haystack.utils import ComponentDevice, Secret


class TestTransformersZeroShotTextRouter:
    def test_to_dict(self):
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        router_dict = router.to_dict()
        assert router_dict == {
            "type": "haystack.components.routers.zero_shot_text_router.TransformersZeroShotTextRouter",
            "init_parameters": {
                "labels": ["query", "passage"],
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.zero_shot_text_router.TransformersZeroShotTextRouter",
            "init_parameters": {
                "labels": ["query", "passage"],
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersZeroShotTextRouter.from_dict(data)
        assert component.labels == ["query", "passage"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "zero-shot-classification",
            "token": None,
        }

    def test_from_dict_no_default_parameters(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        data = {
            "type": "haystack.components.routers.zero_shot_text_router.TransformersZeroShotTextRouter",
            "init_parameters": {"labels": ["query", "passage"]},
        }
        component = TransformersZeroShotTextRouter.from_dict(data)
        assert component.labels == ["query", "passage"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "zero-shot-classification",
            "token": None,
        }

    @patch("haystack.components.routers.zero_shot_text_router.pipeline")
    def test_warm_up(self, hf_pipeline_mock):
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        router.warm_up()
        assert router.pipeline is not None

    def test_run_fails_without_warm_up(self):
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        with pytest.raises(RuntimeError):
            router.run(text="test")

    @patch("haystack.components.routers.zero_shot_text_router.pipeline")
    def test_run_fails_with_non_string_input(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = " "
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        router.warm_up()
        with pytest.raises(TypeError):
            router.run(text=["wrong_input"])

    @patch("haystack.components.routers.zero_shot_text_router.pipeline")
    def test_run_unit(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = [
            {"sequence": "What is the color of the sky?", "labels": ["query", "passage"], "scores": [0.9, 0.1]}
        ]
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        router.pipeline = hf_pipeline_mock
        out = router.run("What is the color of the sky?")
        assert router.pipeline is not None
        assert out == {"query": "What is the color of the sky?"}

    @pytest.mark.integration
    def test_run(self):
        router = TransformersZeroShotTextRouter(labels=["query", "passage"])
        router.warm_up()
        out = router.run("What is the color of the sky?")
        assert router.pipeline is not None
        assert out == {"query": "What is the color of the sky?"}
