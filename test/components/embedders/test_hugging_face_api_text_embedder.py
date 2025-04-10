# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import asyncio
from unittest.mock import MagicMock, patch

import random
import pytest
from huggingface_hub.utils import RepositoryNotFoundError
from numpy import array
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils.auth import Secret
from haystack.utils.hf import HFEmbeddingAPIType
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores import InMemoryDocumentStore


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

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_serverless(self):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])


class TestHuggingFaceAPITextEmbedderAsync:
    """
    Integration tests for HuggingFaceAPITextEmbedder that verify the async functionality with a real API.
    These tests require a valid Hugging Face API token.
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(os.environ.get("HF_API_TOKEN", "") == "", reason="HF_API_TOKEN is not set")
    async def test_run_async_with_real_api(self):
        """
        Integration test that verifies the async functionality with a real API.
        This test requires a valid Hugging Face API token.
        """
        # Use a small, reliable model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model_name}
        )

        # Test with a simple text
        text = "This is a test sentence for embedding."
        result = await embedder.run_async(text=text)

        # Verify the result
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) == 384  # MiniLM-L6-v2 has 384 dimensions

        # Test with a longer text
        long_text = "This is a longer test sentence for embedding. " * 10
        result = await embedder.run_async(text=long_text)

        # Verify the result
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) == 384

        # Test with prefix and suffix
        embedder_with_prefix_suffix = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": model_name},
            prefix="prefix: ",
            suffix=" :suffix",
        )

        result = await embedder_with_prefix_suffix.run_async(text=text)

        # Verify the result
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) == 384

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(os.environ.get("HF_API_TOKEN", "") == "", reason="HF_API_TOKEN is not set")
    async def test_run_async_concurrent_requests(self):
        """
        Integration test that verifies the async functionality with concurrent requests.
        This test requires a valid Hugging Face API token.
        """
        # Use a small, reliable model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model_name}
        )

        # Test with multiple concurrent requests
        texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence.",
            "This is the fourth test sentence.",
            "This is the fifth test sentence.",
        ]

        # Run concurrent requests
        tasks = [embedder.run_async(text=text) for text in texts]
        results = await asyncio.gather(*tasks)

        # Verify the results
        for i, result in enumerate(results):
            assert "embedding" in result
            assert isinstance(result["embedding"], list)
            assert all(isinstance(x, float) for x in result["embedding"])
            assert len(result["embedding"]) == 384  # MiniLM-L6-v2 has 384 dimensions

            # Verify that the embeddings are different
            if i > 0:
                prev_embedding = results[i - 1]["embedding"]
                curr_embedding = result["embedding"]
                assert prev_embedding != curr_embedding  # Different texts should have different embeddings

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(os.environ.get("HF_API_TOKEN", "") == "", reason="HF_API_TOKEN is not set")
    async def test_run_async_error_handling_with_real_api(self):
        """
        Integration test that verifies error handling with a real API.
        This test requires a valid Hugging Face API token.
        """
        # Use an invalid model name to trigger an error
        invalid_model_name = "invalid-model-name-that-does-not-exist"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": invalid_model_name}
        )

        # Test with a simple text
        text = "This is a test sentence for embedding."

        # The request should fail with an appropriate error
        with pytest.raises(Exception) as excinfo:
            await embedder.run_async(text=text)

        # Verify that the error message contains information about the invalid model
        assert invalid_model_name in str(excinfo.value) or "model" in str(excinfo.value).lower()
