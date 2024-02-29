import pytest
from unittest.mock import patch
from haystack.components.routers.zero_shot_text_router import ZeroShotTextRouter
from haystack.utils import ComponentDevice, Secret


class TestFileTypeRouter:
    def test_to_dict(self):
        router = ZeroShotTextRouter(labels=["query", "passage"])
        router_dict = router.to_dict()
        assert router_dict == {
            "type": "haystack.components.routers.zero_shot_text_router.ZeroShotTextRouter",
            "init_parameters": {
                "labels": ["query", "passage"],
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "pipeline_kwargs": {
                    "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.routers.zero_shot_text_router.ZeroShotTextRouter",
            "init_parameters": {
                "labels": ["query", "passage"],
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "pipeline_kwargs": {
                    "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = ZeroShotTextRouter.from_dict(data)
        assert component.labels == ["query", "passage"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict({"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"})
        assert component.pipeline_kwargs == {
            "model": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "zero-shot-classification",
            "token": None,
        }

    def test_run_error(self):
        router = ZeroShotTextRouter(labels=["query", "passage"])
        with pytest.raises(RuntimeError):
            router.run(text="test")

    @patch("haystack.components.routers.zero_shot_text_router.pipeline")
    def test_run_not_str_error(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = " "
        router = ZeroShotTextRouter(labels=["query", "passage"])
        router.warm_up()
        with pytest.raises(TypeError):
            router.run(text=["wrong_input"])

    @pytest.mark.integration
    def test_run(self):
        router = ZeroShotTextRouter(labels=["query", "passage"])
        router.warm_up()
        out = router.run("What is the color of the sky?")
        assert router.pipeline is not None
        assert out == {"query": "What is the color of the sky?"}
