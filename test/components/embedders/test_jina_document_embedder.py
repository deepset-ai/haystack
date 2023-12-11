from unittest.mock import patch

import pytest
import json
import requests

from haystack import Document
from haystack.components.embedders.jina_document_embedder import JinaDocumentEmbedder


def mock_session_post_response(*args, **kwargs):
    inputs = kwargs['json']['input']
    model = kwargs['json']['model']
    mock_response = requests.Response()
    mock_response.status_code = 200
    data = [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
    mock_response._content = json.dumps({"model": model, "object": "list",
                                         "usage": {"total_tokens": 4, "prompt_tokens": 4},
                                         "data": data}).encode()

    return mock_response


class TestJinaDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "fake-api-key")
        embedder = JinaDocumentEmbedder()

        assert embedder.model_name == "jina-embeddings-v2-base-en"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = JinaDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="JinaDocumentEmbedder expects a Jina API key"):
            JinaDocumentEmbedder()

    def test_to_dict(self):
        component = JinaDocumentEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.jina_document_embedder.JinaDocumentEmbedder",
            "init_parameters": {
                "model_name": "jina-embeddings-v2-base-en",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = JinaDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.jina_document_embedder.JinaDocumentEmbedder",
            "init_parameters": {
                "model_name": "model",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = JinaDocumentEmbedder(
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

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = JinaDocumentEmbedder(api_key="fake-api-key", prefix="my_prefix ", suffix=" my_suffix")

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_embed_batch(self):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch('requests.sessions.Session.post', side_effect=mock_session_post_response):
            embedder = JinaDocumentEmbedder(api_key="fake-api-key", model_name="model")

            embeddings, metadata = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 3
            assert all(isinstance(x, float) for x in embedding)

        assert metadata == {"model": "model", "usage": {"prompt_tokens": 3 * 4, "total_tokens": 3 * 4}}

    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "jina-embeddings-v2-base-en"
        with patch('requests.sessions.Session.post', side_effect=mock_session_post_response):
            embedder = JinaDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["metadata"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)
        assert metadata == {"model": model, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    def test_run_custom_batch_size(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]
        model = "jina-embeddings-v2-base-en"
        with patch('requests.sessions.Session.post', side_effect=mock_session_post_response):
            embedder = JinaDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["metadata"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)

        assert metadata == {"model": model, "usage": {"prompt_tokens": 2 * 4, "total_tokens": 2 * 4}}

    def test_run_wrong_input_format(self):
        embedder = JinaDocumentEmbedder(api_key="fake-api-key")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="JinaDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="JinaDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self):
        embedder = JinaDocumentEmbedder(api_key="fake-api-key")

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list
