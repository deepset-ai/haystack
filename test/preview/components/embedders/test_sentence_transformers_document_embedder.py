from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from haystack.preview import Document
from haystack.preview.components.embedders.sentence_transformers_document_embedder import (
    SentenceTransformersDocumentEmbedder,
)


class TestSentenceTransformersDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
        assert embedder.model_name_or_path == "model"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is None
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = SentenceTransformersDocumentEmbedder(
            model_name_or_path="model",
            device="cuda",
            use_auth_token=True,
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model_name_or_path == "model"
        assert embedder.device == "cuda"
        assert embedder.use_auth_token is True
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_to_dict(self):
        component = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
        data = component.to_dict()
        assert data == {
            "type": "SentenceTransformersDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cpu",
                "use_auth_token": None,
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "embedding_separator": "\n",
                "metadata_fields_to_embed": [],
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersDocumentEmbedder(
            model_name_or_path="model",
            device="cuda",
            use_auth_token="the-token",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            metadata_fields_to_embed=["meta_field"],
            embedding_separator=" - ",
        )
        data = component.to_dict()
        assert data == {
            "type": "SentenceTransformersDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cuda",
                "use_auth_token": "the-token",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "embedding_separator": " - ",
                "metadata_fields_to_embed": ["meta_field"],
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "SentenceTransformersDocumentEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cuda",
                "use_auth_token": "the-token",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": False,
                "embedding_separator": " - ",
                "metadata_fields_to_embed": ["meta_field"],
            },
        }
        component = SentenceTransformersDocumentEmbedder.from_dict(data)
        assert component.model_name_or_path == "model"
        assert component.device == "cuda"
        assert component.use_auth_token == "the-token"
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.normalize_embeddings is False
        assert component.metadata_fields_to_embed == ["meta_field"]
        assert component.embedding_separator == " - "

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name_or_path="model", device="cpu", use_auth_token=None
        )

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    @pytest.mark.unit
    def test_run(self):
        embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        documents = [Document(content=f"document number {i}") for i in range(5)]

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersDocumentEmbedder(model_name_or_path="model")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_embed_metadata(self):
        embedder = SentenceTransformersDocumentEmbedder(
            model_name_or_path="model", metadata_fields_to_embed=["meta_field"], embedding_separator="\n"
        )
        embedder.embedding_backend = MagicMock()

        documents = [
            Document(content=f"document number {i}", metadata={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder.run(documents=documents)

        embedder.embedding_backend.embed.assert_called_once_with(
            [
                "meta_value 0\ndocument number 0",
                "meta_value 1\ndocument number 1",
                "meta_value 2\ndocument number 2",
                "meta_value 3\ndocument number 3",
                "meta_value 4\ndocument number 4",
            ],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
