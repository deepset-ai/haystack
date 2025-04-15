# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

from openai import APIError

from haystack.utils.auth import Secret
import pytest
import httpx

from haystack import Document
from haystack.components.embedders import AzureOpenAIDocumentEmbedder
from haystack.utils.azure import default_azure_ad_token_provider
from unittest.mock import Mock, patch
from haystack.utils.http_client import init_http_client


class TestAzureOpenAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAIDocumentEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.dimensions is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.default_headers == {}
        assert embedder.azure_ad_token_provider is None
        assert embedder.http_client_kwargs is None

    def test_init_with_0_max_retries(self, monkeypatch):
        """Tests that the max_retries init param is set correctly if equal 0"""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAIDocumentEmbedder(
            azure_endpoint="https://example-resource.azure.openai.com/", max_retries=0
        )
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.dimensions is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.default_headers == {}
        assert embedder.azure_ad_token_provider is None
        assert embedder.max_retries == 0

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        component = AzureOpenAIDocumentEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.azure_document_embedder.AzureOpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": None,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "max_retries": 5,
                "timeout": 30.0,
                "default_headers": {},
                "azure_ad_token_provider": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        component = AzureOpenAIDocumentEmbedder(
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
            "type": "haystack.components.embedders.azure_document_embedder.AzureOpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": 768,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "organization": "HaystackCI",
                "prefix": "prefix ",
                "suffix": " suffix",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "max_retries": 10,
                "timeout": 60.0,
                "default_headers": {"x-custom-header": "custom-value"},
                "azure_ad_token_provider": "haystack.utils.azure.default_azure_ad_token_provider",
                "http_client_kwargs": {"proxy": "http://example.com:3128", "verify": False},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.embedders.azure_document_embedder.AzureOpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": None,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "max_retries": 5,
                "timeout": 30.0,
                "default_headers": {},
                "azure_ad_token_provider": None,
                "http_client_kwargs": None,
            },
        }
        component = AzureOpenAIDocumentEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
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
            "type": "haystack.components.embedders.azure_document_embedder.AzureOpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-ada-002",
                "dimensions": 768,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "organization": "HaystackCI",
                "prefix": "prefix ",
                "suffix": " suffix",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "max_retries": 10,
                "timeout": 60.0,
                "default_headers": {"x-custom-header": "custom-value"},
                "azure_ad_token_provider": "haystack.utils.azure.default_azure_ad_token_provider",
                "http_client_kwargs": {"proxy": "http://example.com:3128", "verify": False},
            },
        }
        component = AzureOpenAIDocumentEmbedder.from_dict(data)
        assert component.azure_deployment == "text-embedding-ada-002"
        assert component.azure_endpoint == "https://example-resource.azure.openai.com/"
        assert component.api_version == "2023-05-15"
        assert component.max_retries == 10
        assert component.timeout == 60.0
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"
        assert component.default_headers == {"x-custom-header": "custom-value"}
        assert component.azure_ad_token_provider is not None
        assert component.http_client_kwargs == {"proxy": "http://example.com:3128", "verify": False}

    def test_embed_batch_handles_exceptions_gracefully(self, caplog):
        embedder = AzureOpenAIDocumentEmbedder(
            azure_endpoint="https://test.openai.azure.com",
            api_key=Secret.from_token("fake-api-key"),
            azure_deployment="text-embedding-ada-002",
            embedding_separator=" | ",
        )

        fake_texts_to_embed = {"1": "text1", "2": "text2"}

        with patch.object(
            embedder.client.embeddings,
            "create",
            side_effect=APIError(message="Mocked error", request=Mock(), body=None),
        ):
            embedder._embed_batch(texts_to_embed=fake_texts_to_embed, batch_size=32)

        assert len(caplog.records) == 1
        assert "Failed embedding of documents 1, 2 caused by Mocked error" in caplog.text

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
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]
        # the default model is text-embedding-ada-002 even if we don't specify it, but let's be explicit
        embedder = AzureOpenAIDocumentEmbedder(
            azure_deployment="text-embedding-ada-002",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
            organization="HaystackCI",
        )

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]
        metadata = result["meta"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1536
            assert all(isinstance(x, float) for x in doc.embedding)

        assert metadata["usage"]["prompt_tokens"] == 15
        assert metadata["usage"]["total_tokens"] == 15
        assert "ada" in metadata["model"]
