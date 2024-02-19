from unittest.mock import patch, MagicMock
import pytest
from haystack.utils import Secret, ComponentDevice

import numpy as np

from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder


class TestSentenceTransformersTextEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    def test_init_with_parameters(self):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    def test_to_dict(self):
        component = SentenceTransformersTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cpu")
        assert component.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model", token=None, device=ComponentDevice.from_str("cpu"))
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(model="model", device="cpu", auth_token=None)

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        text = "a nice text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(el, float) for el in embedding)

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)
