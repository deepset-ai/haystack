from unittest.mock import patch, MagicMock
import pytest

from haystack.preview.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder

from test.preview.components.base import BaseTestComponent

import numpy as np


class TestSentenceTransformersTextEmbedder(BaseTestComponent):
    # TODO: We're going to rework these tests when we'll remove BaseTestComponent.

    @pytest.mark.unit
    def test_init_default(self):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        assert embedder.model_name_or_path == "model"
        assert embedder.device is None
        assert embedder.use_auth_token is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = SentenceTransformersTextEmbedder(
            model_name_or_path="model",
            device="cpu",
            use_auth_token=True,
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        assert embedder.model_name_or_path == "model"
        assert embedder.device == "cpu"
        assert embedder.use_auth_token is True
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_text_embedder.SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model_name_or_path="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model_name_or_path="model", device=None, use_auth_token=None
        )

    @pytest.mark.unit
    @patch(
        "haystack.preview.components.embedders.sentence_transformers_text_embedder.SentenceTransformersEmbeddingBackendFactory"
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
        embedder.embedding_backend.embed = lambda x, **kwargs: list(np.random.rand(len(x), 16))

        texts = ["sentence1", "sentence2"]

        result = embedder.run(texts=texts)

        embeddings = result["embeddings"]
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
