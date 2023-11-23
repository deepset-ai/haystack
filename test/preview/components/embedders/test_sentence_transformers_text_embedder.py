from unittest.mock import patch, MagicMock
import pytest

import numpy as np

from haystack.preview.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder


class TestSentenceTransformersTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        assert embedder.model_name_or_path == "model"
        assert embedder.device == "cpu"
        assert embedder.token is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = SentenceTransformersTextEmbedder(
            model_name_or_path="model",
            device="cuda",
            token=True,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        assert embedder.model_name_or_path == "model"
        assert embedder.device == "cuda"
        assert embedder.token is True
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    @pytest.mark.unit
    def test_to_dict(self):
        component = SentenceTransformersTextEmbedder(model_name_or_path="model")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cpu",
                "token": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersTextEmbedder(
            model_name_or_path="model",
            device="cuda",
            token=True,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cuda",
                "token": True,
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }

    @pytest.mark.unit
    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersTextEmbedder(model_name_or_path="model", token="awesome-token")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "model_name_or_path": "model",
                "device": "cpu",
                "token": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name_or_path="model", device="cpu", use_auth_token=None
        )

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    @pytest.mark.unit
    def test_run(self):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        text = "a nice text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(el, float) for el in embedding)

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)
