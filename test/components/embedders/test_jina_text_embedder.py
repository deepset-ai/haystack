from unittest.mock import patch
import pytest
import json
import requests
from haystack.components.embedders.jina_text_embedder import JinaTextEmbedder


class TestJinaTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaTextEmbedder()

        assert embedder.model_name == "jina-embeddings-v2-base-en"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = JinaTextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="JinaTextEmbedder expects a Jina API key"):
            JinaTextEmbedder()

    def test_to_dict(self):
        component = JinaTextEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.jina_text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "model_name": "jina-embeddings-v2-base-en",
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = JinaTextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.jina_text_embedder.JinaTextEmbedder",
            "init_parameters": {
                "model_name": "model",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    def test_run(self):
        model = "jina-embeddings-v2-base-en"
        with patch('requests.sessions.Session.post') as mock_post:
            # Configure the mock to return a specific response
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_response._content = json.dumps({"model": "jina-embeddings-v2-base-en", "object": "list",
                                                 "usage": {"total_tokens": 6, "prompt_tokens": 6},
                                                 "data": [{"object": "embedding", "index": 0,
                                                           "embedding": [0.1, 0.2, 0.3]}]}).encode()

            mock_post.return_value = mock_response

            embedder = JinaTextEmbedder(api_key="fake-api-key", model_name=model, prefix="prefix ", suffix=" suffix")
            result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["metadata"] == {"model": "jina-embeddings-v2-base-en",
                                      "usage": {"prompt_tokens": 6, "total_tokens": 6}}

    def test_run_wrong_input_format(self):
        embedder = JinaTextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="JinaTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
