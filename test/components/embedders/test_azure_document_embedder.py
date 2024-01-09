import os

import pytest

from haystack import Document
from haystack.components.embedders import AzureOpenAIDocumentEmbedder


class TestAzureOpenAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        embedder = AzureOpenAIDocumentEmbedder(azure_endpoint="https://example-resource.azure.openai.com/")
        assert embedder.azure_deployment == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_to_dict(self):
        component = AzureOpenAIDocumentEmbedder(
            api_key="fake-api-key", azure_endpoint="https://example-resource.azure.openai.com/"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.azure_document_embedder.AzureOpenAIDocumentEmbedder",
            "init_parameters": {
                "api_version": "2023-05-15",
                "azure_deployment": "text-embedding-ada-002",
                "azure_endpoint": "https://example-resource.azure.openai.com/",
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
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
        assert metadata == {"model": "ada", "usage": {"prompt_tokens": 15, "total_tokens": 15}}
