from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from haystack import Document
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder


class TestSentenceTransformersDocumentEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersDocumentEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == "cpu"
        assert embedder.token is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = SentenceTransformersDocumentEmbedder(
            model="model",
            device="cuda",
            token=True,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.model == "model"
        assert embedder.device == "cuda"
        assert embedder.token is True
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self):
        component = SentenceTransformersDocumentEmbedder(model="model")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder",
            "init_parameters": {
                "model": "model",
                "device": "cpu",
                "token": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "embedding_separator": "\n",
                "meta_fields_to_embed": [],
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersDocumentEmbedder(
            model="model",
            device="cuda",
            token="the-token",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" - ",
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder",
            "init_parameters": {
                "model": "model",
                "device": "cuda",
                "token": None,  # the token is not serialized
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "embedding_separator": " - ",
                "meta_fields_to_embed": ["meta_field"],
            },
        }

    @patch(
        "haystack.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersDocumentEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(model="model", device="cpu", use_auth_token=None)

    @patch(
        "haystack.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersDocumentEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersDocumentEmbedder(model="model")
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

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersDocumentEmbedder(model="model")

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

    def test_embed_metadata(self):
        embedder = SentenceTransformersDocumentEmbedder(
            model="model", meta_fields_to_embed=["meta_field"], embedding_separator="\n"
        )
        embedder.embedding_backend = MagicMock()

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

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

    def test_prefix_suffix(self):
        embedder = SentenceTransformersDocumentEmbedder(
            model="model",
            prefix="my_prefix ",
            suffix=" my_suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        embedder.embedding_backend = MagicMock()

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        embedder.run(documents=documents)

        embedder.embedding_backend.embed.assert_called_once_with(
            [
                "my_prefix meta_value 0\ndocument number 0 my_suffix",
                "my_prefix meta_value 1\ndocument number 1 my_suffix",
                "my_prefix meta_value 2\ndocument number 2 my_suffix",
                "my_prefix meta_value 3\ndocument number 3 my_suffix",
                "my_prefix meta_value 4\ndocument number 4 my_suffix",
            ],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
