import os

import pytest
from openai import OpenAIError

from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder


class TestOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAITextEmbedder()

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = OpenAITextEmbedder(
            api_key="fake-api-key", model="model", organization="fake-organization", prefix="prefix", suffix="suffix"
        )
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAIError):
            OpenAITextEmbedder()

    def test_to_dict(self):
        component = OpenAITextEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {"model": "text-embedding-ada-002", "organization": None, "prefix": "", "suffix": ""},
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = OpenAITextEmbedder(
            api_key="fake-api-key", model="model", organization="fake-organization", prefix="prefix", suffix="suffix"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "model": "model",
                "organization": "fake-organization",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    def test_run_wrong_input_format(self):
        embedder = OpenAITextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAITextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "text-embedding-ada-002"

        embedder = OpenAITextEmbedder(model=model, prefix="prefix ", suffix=" suffix")
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1536
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": "text-embedding-ada-002-v2",
            "usage": {"prompt_tokens": 6, "total_tokens": 6},
        }
