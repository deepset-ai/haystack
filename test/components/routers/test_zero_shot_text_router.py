import pytest
from unittest.mock import patch
from haystack.components.routers.zero_shot_text_router import ZeroShotTextRouter


@pytest.fixture
def text_router():
    return ZeroShotTextRouter(labels=["query", "passage"])


class TestFileTypeRouter:
    # def test_to_dict(self):
    #     router = ZeroShotTextRouter(labels=["query", "passage"])
    #     router_dict = router.to_dict()
    #     pass

    # def test_from_dict(self):
    #     pass

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

    # @pytest.mark.integration
    # def test_run(self):
    #     pass
