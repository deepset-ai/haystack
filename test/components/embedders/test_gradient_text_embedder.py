import pytest
from gradientai.openapi.client.models.generate_embedding_success import GenerateEmbeddingSuccess
from haystack.preview.components.embedders.gradient_text_embedder import GradientTextEmbedder
from unittest.mock import MagicMock, NonCallableMagicMock
import numpy as np


access_token = "access_token"
workspace_id = "workspace_id"
model = "bge-large"


class TestGradientTextEmbedder:
    @pytest.mark.unit
    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", access_token)
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", workspace_id)

        embedder = GradientTextEmbedder()
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_without_access_token(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_ACCESS_TOKEN", raising=True)

        with pytest.raises(ValueError):
            GradientTextEmbedder(workspace_id=workspace_id)

    @pytest.mark.unit
    def test_init_without_workspace(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_WORKSPACE_ID", raising=True)

        with pytest.raises(ValueError):
            GradientTextEmbedder(access_token=access_token)

    @pytest.mark.unit
    def test_init_from_params(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_init_from_params_precedence(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", "env_access_token")
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", "env_workspace_id")

        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    @pytest.mark.unit
    def test_to_dict(self):
        component = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        data = component.to_dict()
        assert data == {
            "type": "GradientTextEmbedder",
            "init_parameters": {"workspace_id": workspace_id, "model_name": "bge-large"},
        }

    @pytest.mark.unit
    def test_warmup(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(slug="bge-large")

    @pytest.mark.unit
    def test_warmup_doesnt_reload(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._gradient.get_embeddings_model = MagicMock(default_return_value="fake model")
        embedder.warm_up()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(slug="bge-large")

    @pytest.mark.unit
    def test_run_fail_if_not_warmed_up(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)

        with pytest.raises(RuntimeError, match="warm_up()"):
            embedder.run(text="The food was delicious")

    @pytest.mark.unit
    def test_run_fail_when_no_embeddings_returned(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.generate_embeddings.return_value = GenerateEmbeddingSuccess(embeddings=[])

        with pytest.raises(RuntimeError):
            _result = embedder.run(text="The food was delicious")
            embedder._embedding_model.generate_embeddings.assert_called_once_with(
                inputs=[{"input": "The food was delicious"}]
            )

    @pytest.mark.unit
    def test_run_empty_string(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.generate_embeddings.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": 0}]
        )

        result = embedder.run(text="")
        embedder._embedding_model.generate_embeddings.assert_called_once_with(inputs=[{"input": ""}])

        assert len(result["embedding"]) == 1024  # 1024 is the bge-large embedding size
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.unit
    def test_run(self):
        embedder = GradientTextEmbedder(access_token=access_token, workspace_id=workspace_id)
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.generate_embeddings.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": 0}]
        )

        result = embedder.run(text="The food was delicious")
        embedder._embedding_model.generate_embeddings.assert_called_once_with(
            inputs=[{"input": "The food was delicious"}]
        )

        assert len(result["embedding"]) == 1024  # 1024 is the bge-large embedding size
        assert all(isinstance(x, float) for x in result["embedding"])
