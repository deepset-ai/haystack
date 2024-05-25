# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.utils.auth import Secret


class TestOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAITextEmbedder()

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.api_base_url == None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.client.timeout == 30
        assert embedder.client.max_retries == 5

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
            timeout=40.0,
            max_retries=1,
        )
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.api_base_url == "https://my-custom-base-url.com"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.client.timeout == 40.0
        assert embedder.client.max_retries == 1

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.api_base_url == "https://my-custom-base-url.com"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.client.timeout == 100.0
        assert embedder.client.max_retries == 10

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            OpenAITextEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        component = OpenAITextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": None,
                "dimensions": None,
                "model": "text-embedding-ada-002",
                "organization": None,
                "prefix": "",
                "suffix": "",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = OpenAITextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "api_base_url": "https://my-custom-base-url.com",
                "model": "model",
                "dimensions": None,
                "organization": "fake-organization",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    def test_run_wrong_input_format(self):
        embedder = OpenAITextEmbedder(api_key=Secret.from_token("fake-api-key"))

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

        assert (
            "text" in result["meta"]["model"] and "ada" in result["meta"]["model"]
        ), "The model name does not contain 'text' and 'ada'"

        assert result["meta"]["usage"] == {"prompt_tokens": 6, "total_tokens": 6}, "Usage information does not match"
