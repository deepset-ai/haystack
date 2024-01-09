import os

import pytest

from haystack.components.embedders import AzureOpenAITextEmbedder


class TestAzureOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAITextEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")

        assert embedder._client.api_key == "fake-api-key"
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_to_dict(self):
        component = AzureOpenAITextEmbedder(
            api_key="fake-api-key", azure_endpoint="https://example-resource.azure.openai.com/"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.azure_text_embedder.AzureOpenAITextEmbedder",
            "init_parameters": {
                "azure_deployment": "text-embedding-ada-002",
                "organization": None,
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "api_version": "2023-05-15",
                "prefix": "",
                "suffix": "",
            },
        }

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
        assert result["meta"] == {"model": "ada", "usage": {"prompt_tokens": 6, "total_tokens": 6}}
