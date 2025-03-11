# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from haystack.components.embedders import AzureOpenAITextEmbedder
from haystack.utils.azure import default_azure_ad_token_provider


class TestAzureOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAITextEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")

        assert embedder._client.api_key == "fake-api-key"
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.dimensions is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.default_headers == {}
        assert embedder.azure_ad_token_provider is None

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        component = AzureOpenAITextEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.azure_text_embedder.AzureOpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": None,
                "organization": None,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "api_version": "2023-05-15",
                "max_retries": 5,
                "timeout": 30.0,
                "prefix": "",
                "suffix": "",
                "default_headers": {},
                "azure_ad_token_provider": None,
            },
        }

    def test_to_dict_with_params(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        component = AzureOpenAITextEmbedder(
            azure_endpoint="https://example-resource.azure.openai.com/",
            azure_deployment="text-embedding-ada-002",
            dimensions=768,
            organization="HaystackCI",
            timeout=60.0,
            max_retries=10,
            prefix="prefix ",
            suffix=" suffix",
            default_headers={"x-custom-header": "custom-value"},
            azure_ad_token_provider=default_azure_ad_token_provider,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.azure_text_embedder.AzureOpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": 768,
                "organization": "HaystackCI",
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "api_version": "2023-05-15",
                "max_retries": 10,
                "timeout": 60.0,
                "prefix": "prefix ",
                "suffix": " suffix",
                "default_headers": {"x-custom-header": "custom-value"},
                "azure_ad_token_provider": "haystack.utils.azure.default_azure_ad_token_provider",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.embedders.azure_text_embedder.AzureOpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": None,
                "organization": None,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "api_version": "2023-05-15",
                "max_retries": 5,
                "timeout": 30.0,
                "prefix": "",
                "suffix": "",
                "default_headers": {},
            },
        }
        component = AzureOpenAITextEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
        assert component.azure_endpoint == "https://example-resource.azure.openai.com/"
        assert component.api_version == "2023-05-15"
        assert component.max_retries == 5
        assert component.timeout == 30.0
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.default_headers == {}
        assert component.azure_ad_token_provider is None

    def test_from_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.embedders.azure_text_embedder.AzureOpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": 768,
                "organization": "HaystackCI",
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "api_version": "2023-05-15",
                "max_retries": 10,
                "timeout": 60.0,
                "prefix": "prefix ",
                "suffix": " suffix",
                "default_headers": {"x-custom-header": "custom-value"},
                "azure_ad_token_provider": "haystack.utils.azure.default_azure_ad_token_provider",
            },
        }
        component = AzureOpenAITextEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
        assert component.azure_endpoint == "https://example-resource.azure.openai.com/"
        assert component.api_version == "2023-05-15"
        assert component.max_retries == 10
        assert component.timeout == 60.0
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"
        assert component.default_headers == {"x-custom-header": "custom-value"}
        assert component.azure_ad_token_provider is not None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) and not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    def test_run(self):
        # the default model is text-embedding-ada-002 even if we don't specify it, but let's be explicit
        embedder = AzureOpenAITextEmbedder(
            azure_deployment="text-embedding-ada-002", prefix="prefix ", suffix=" suffix", organization="HaystackCI"
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1536
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {"model": "text-embedding-ada-002", "usage": {"prompt_tokens": 6, "total_tokens": 6}}
