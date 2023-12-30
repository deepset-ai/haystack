from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.embedders.hugging_face_tei_document_embedder import HuggingFaceTEIDocumentEmbedder
from haystack.dataclasses import Document


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.embedders.hugging_face_tei_document_embedder.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


def mock_embedding_generation(text, **kwargs):
    response = np.array([np.random.rand(384) for i in range(len(text))])
    return response


class TestHuggingFaceTEIDocumentEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model):
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = HuggingFaceTEIDocumentEmbedder()

        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.url is None
        assert embedder.token == "fake-api-token"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self, mock_check_valid_model):
        embedder = HuggingFaceTEIDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token="fake-api-token",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.url == "https://some_embedding_model.com"
        assert embedder.token == "fake-api-token"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_initialize_with_invalid_url(self, mock_check_valid_model):
        with pytest.raises(ValueError):
            HuggingFaceTEIDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2", url="invalid_url")

    def test_initialize_with_url_but_invalid_model(self, mock_check_valid_model):
        # When custom TEI endpoint is used via URL, model must be provided and valid HuggingFace Hub model id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceTEIDocumentEmbedder(model="invalid_model_id", url="https://some_embedding_model.com")

    def test_to_dict(self, mock_check_valid_model):
        component = HuggingFaceTEIDocumentEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_tei_document_embedder.HuggingFaceTEIDocumentEmbedder",
            "init_parameters": {
                "model": "BAAI/bge-small-en-v1.5",
                "url": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, mock_check_valid_model):
        component = HuggingFaceTEIDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token="fake-api-token",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )

        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_tei_document_embedder.HuggingFaceTEIDocumentEmbedder",
            "init_parameters": {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "url": "https://some_embedding_model.com",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self, mock_check_valid_model):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = HuggingFaceTEIDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token="fake-api-token",
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_prepare_texts_to_embed_w_suffix(self, mock_check_valid_model):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = HuggingFaceTEIDocumentEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token="fake-api-token",
            prefix="my_prefix ",
            suffix=" my_suffix",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_embed_batch(self, mock_check_valid_model):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceTEIDocumentEmbedder(
                model="BAAI/bge-small-en-v1.5", url="https://some_embedding_model.com", token="fake-api-token"
            )
            embeddings = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

            assert mock_embedding_patch.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    def test_run(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceTEIDocumentEmbedder(
                model="BAAI/bge-small-en-v1.5",
                token="fake-api-token",
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

            mock_embedding_patch.assert_called_once_with(
                text=[
                    "prefix Cuisine | I love cheese suffix",
                    "prefix ML | A transformer is a deep learning architecture suffix",
                ]
            )
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    def test_run_custom_batch_size(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceTEIDocumentEmbedder(
                model="BAAI/bge-small-en-v1.5",
                token="fake-api-token",
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

            assert mock_embedding_patch.call_count == 2

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceTEIDocumentEmbedder(
            model="BAAI/bge-small-en-v1.5", url="https://some_embedding_model.com", token="fake-api-token"
        )

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="HuggingFaceTEIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="HuggingFaceTEIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self, mock_check_valid_model):
        embedder = HuggingFaceTEIDocumentEmbedder(
            model="BAAI/bge-small-en-v1.5", url="https://some_embedding_model.com", token="fake-api-token"
        )

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list
