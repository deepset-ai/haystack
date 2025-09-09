# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import sys
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.utils import RepositoryNotFoundError
from numpy import array

from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils.auth import Secret
from haystack.utils.hf import HFEmbeddingAPIType


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.embedders.hugging_face_api_text_embedder.check_valid_model", MagicMock(return_value=None)
    ) as mock:
        yield mock


class TestHuggingFaceAPITextEmbedder:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self, mock_check_valid_model):
        model = "BAAI/bge-small-en-v1.5"
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model}
        )

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": model}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize

    def test_init_serverless_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
            )

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tei(self):
        url = "https://some_model.com"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": url}
        )

        assert embedder.api_type == HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        assert embedder.api_params == {"url": url}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize

    def test_init_tei_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tei_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_to_dict(self, mock_check_valid_model):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
        )

        data = embedder.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder",
            "init_parameters": {
                "api_type": "serverless_inference_api",
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

    def test_from_dict(self, mock_check_valid_model):
        data = {
            "type": "haystack.components.embedders.hugging_face_api_text_embedder.HuggingFaceAPITextEmbedder",
            "init_parameters": {
                "api_type": HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

        embedder = HuggingFaceAPITextEmbedder.from_dict(data)

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": "BAAI/bge-small-en-v1.5"}
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.truncate
        assert embedder.normalize

    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    def test_run(self, mock_check_valid_model, caplog):
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[random.random() for _ in range(384)]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
            )

            result = embedder.run(text="The food was delicious")

            mock_embedding_patch.assert_called_once_with(
                text="prefix The food was delicious suffix", truncate=None, normalize=None
            )

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

        # Check that warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    @pytest.mark.asyncio
    async def test_run_async(self, mock_check_valid_model, caplog):
        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[random.random() for _ in range(384)]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
            )

            result = await embedder.run_async(text="The food was delicious")

            mock_embedding_patch.assert_called_once_with(
                text="prefix The food was delicious suffix", truncate=None, normalize=None
            )

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

        # Check that warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    def test_run_wrong_embedding_shape(self, mock_check_valid_model):
        # embedding ndim > 2
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
            )

            with pytest.raises(ValueError):
                embedder.run(text="The food was delicious")

        # embedding ndim == 2 but shape[0] != 1
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
            )

            with pytest.raises(ValueError):
                embedder.run(text="The food was delicious")

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=3, reruns_delay=10)
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    @pytest.mark.skipif(sys.platform != "linux", reason="We only test on Linux to avoid overloading the HF server")
    def test_live_run_serverless(self):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
        )
        embedder._client.timeout = 10  # we want to fail fast if the server is not responding
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=3, reruns_delay=10)
    @pytest.mark.skipif(os.environ.get("HF_API_TOKEN", "") == "", reason="HF_API_TOKEN is not set")
    @pytest.mark.skipif(sys.platform != "linux", reason="We only test on Linux to avoid overloading the HF server")
    async def test_live_run_async_serverless(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model_name}
        )
        embedder._client.timeout = 10  # we want to fail fast if the server is not responding

        text = "This is a test sentence for embedding."
        result = await embedder.run_async(text=text)

        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) == 384  # MiniLM-L6-v2 has 384 dimensions
