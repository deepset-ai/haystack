import os
from typing import List

import numpy as np
import pytest
from openai import OpenAIError

from haystack import Document
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder


def mock_openai_response(input: List[str], model: str = "text-embedding-ada-002", **kwargs) -> dict:
    dict_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": np.random.rand(1536).tolist()} for i in range(len(input))
        ],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    return dict_response


class TestOpenAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAIDocumentEmbedder()
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = OpenAIDocumentEmbedder(
            api_key="fake-api-key",
            model="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.organization == "my-org"
        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAIError):
            OpenAIDocumentEmbedder()

    def test_to_dict(self):
        component = OpenAIDocumentEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "api_base_url": None,
                "model": "text-embedding-ada-002",
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = OpenAIDocumentEmbedder(
            api_key="fake-api-key",
            model="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "api_base_url": None,
                "model": "model",
                "organization": "my-org",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = OpenAIDocumentEmbedder(
            api_key="fake-api-key", meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        # note that newline is replaced by space
        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = OpenAIDocumentEmbedder(api_key="fake-api-key", prefix="my_prefix ", suffix=" my_suffix")

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_run_wrong_input_format(self):
        embedder = OpenAIDocumentEmbedder(api_key="fake-api-key")

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self):
        embedder = OpenAIDocumentEmbedder(api_key="fake-api-key")

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-embedding-ada-002"

        embedder = OpenAIDocumentEmbedder(model=model, meta_fields_to_embed=["topic"], embedding_separator=" | ")

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
        assert metadata == {"model": "text-embedding-ada-002-v2", "usage": {"prompt_tokens": 15, "total_tokens": 15}}
