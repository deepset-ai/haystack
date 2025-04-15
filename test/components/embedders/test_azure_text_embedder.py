# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import httpx
import pytest

from haystack.components.embedders import AzureOpenAITextEmbedder
from haystack.utils.azure import default_azure_ad_token_provider
from haystack.utils.http_client import init_http_client


class TestAzureOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAITextEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.dimensions is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.default_headers == {}
        assert embedder.azure_ad_token_provider is None
        assert embedder.http_client_kwargs is None

    def test_init_with_zero_max_retries(self, monkeypatch):
        """Tests that the max_retries init param is set correctly if equal 0"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAITextEmbedder(azure_endpoint="https://example-resource.azure.openai.com/", max_retries=0)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.dimensions is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.default_headers == {}
        assert embedder.azure_ad_token_provider is None
        assert embedder.max_retries == 0

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
                "http_client_kwargs": None,
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
            http_client_kwargs={"proxy": "http://example.com:3128", "verify": False},
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
                "http_client_kwargs": {"proxy": "http://example.com:3128", "verify": False},
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
                "http_client_kwargs": None,
            },
        }
        component = AzureOpenAITextEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
        assert component.model == "text-embedding-ada-002"
        assert component.azure_endpoint == "https://example-resource.azure.openai.com/"
        assert component.api_version == "2023-05-15"
        assert component.max_retries == 5
        assert component.timeout == 30.0
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.default_headers == {}
        assert component.azure_ad_token_provider is None
        assert component.http_client_kwargs is None

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
                "http_client_kwargs": {"proxy": "http://example.com:3128", "verify": False},
            },
        }
        component = AzureOpenAITextEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
        assert component.model == "text-embedding-ada-002"
        assert component.azure_endpoint == "https://example-resource.azure.openai.com/"
        assert component.api_version == "2023-05-15"
        assert component.max_retries == 10
        assert component.timeout == 60.0
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"
        assert component.default_headers == {"x-custom-header": "custom-value"}
        assert component.azure_ad_token_provider is not None
        assert component.http_client_kwargs == {"proxy": "http://example.com:3128", "verify": False}

    def test_init_http_client(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")

        embedder = AzureOpenAITextEmbedder()
        client = init_http_client(embedder.http_client_kwargs, async_client=False)
        assert client is None

        embedder.http_client_kwargs = {"proxy": "http://example.com:3128"}
        client = init_http_client(embedder.http_client_kwargs, async_client=False)
        assert isinstance(client, httpx.Client)

        client = init_http_client(embedder.http_client_kwargs, async_client=True)
        assert isinstance(client, httpx.AsyncClient)

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
        assert result["meta"]["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
        assert "ada" in result["meta"]["model"]
