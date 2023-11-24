from unittest.mock import patch
from typing import List, cast

import pytest
import numpy as np
import openai
from openai.util import convert_to_openai_object
from openai.openai_object import OpenAIObject

from haystack.preview import Document
from haystack.preview.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder


def mock_openai_response(input: List[str], model: str = "text-embedding-ada-002", **kwargs) -> OpenAIObject:
    dict_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": np.random.rand(1536).tolist()} for i in range(len(input))
        ],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    return cast(OpenAIObject, convert_to_openai_object(dict_response))


class TestOpenAIDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        openai.api_key = None
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAIDocumentEmbedder()

        assert openai.api_key == "fake-api-key"

        assert embedder.model_name == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = OpenAIDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert openai.api_key == "fake-api-key"
        assert openai.organization == "my-org"

        assert embedder.organization == "my-org"
        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        openai.api_key = None
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAIDocumentEmbedder expects an OpenAI API key"):
            OpenAIDocumentEmbedder()

    @pytest.mark.unit
    def test_to_dict(self):
        component = OpenAIDocumentEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "model_name": "text-embedding-ada-002",
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = OpenAIDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "model_name": "model",
                "organization": "my-org",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    @pytest.mark.unit
    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = OpenAIDocumentEmbedder(
            api_key="fake-api-key", metadata_fields_to_embed=["meta_field"], embedding_separator=" | "
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_embed_batch(self):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch(
            "haystack.preview.components.embedders.openai_document_embedder.openai.Embedding"
        ) as openai_embedding_patch:
            openai_embedding_patch.create.side_effect = mock_openai_response
            embedder = OpenAIDocumentEmbedder(api_key="fake-api-key", model_name="model")

            embeddings, metadata = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

            assert openai_embedding_patch.create.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)

        # openai.Embedding.create is called 3 times
        assert metadata == {"model": "model", "usage": {"prompt_tokens": 3 * 4, "total_tokens": 3 * 4}}

    @pytest.mark.unit
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-similarity-ada-001"
        with patch(
            "haystack.preview.components.embedders.openai_document_embedder.openai.Embedding"
        ) as openai_embedding_patch:
            openai_embedding_patch.create.side_effect = mock_openai_response
            embedder = OpenAIDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

            openai_embedding_patch.create.assert_called_once_with(
                model=model,
                input=[
                    "prefix Cuisine | I love cheese suffix",
                    "prefix ML | A transformer is a deep learning architecture suffix",
                ],
            )
        documents_with_embeddings = result["documents"]
        metadata = result["metadata"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1536
            assert all(isinstance(x, float) for x in doc.embedding)
        assert metadata == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    @pytest.mark.unit
    def test_run_custom_batch_size(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-similarity-ada-001"
        with patch(
            "haystack.preview.components.embedders.openai_document_embedder.openai.Embedding"
        ) as openai_embedding_patch:
            openai_embedding_patch.create.side_effect = mock_openai_response
            embedder = OpenAIDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

            assert openai_embedding_patch.create.call_count == 2

        documents_with_embeddings = result["documents"]
        metadata = result["metadata"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1536
            assert all(isinstance(x, float) for x in doc.embedding)

        # openai.Embedding.create is called 2 times
        assert metadata == {"model": model, "usage": {"prompt_tokens": 2 * 4, "total_tokens": 2 * 4}}

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = OpenAIDocumentEmbedder(api_key="fake-api-key")

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_run_on_empty_list(self):
        embedder = OpenAIDocumentEmbedder(api_key="fake-api-key")

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list
