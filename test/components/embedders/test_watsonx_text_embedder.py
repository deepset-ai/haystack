# SPDX-FileCopyrightText: 2024-present IBM Corporation
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils.auth import Secret
from haystack.components.embedders.watsonx_text_embedder import WatsonXTextEmbedder
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure


class TestWatsonXTextEmbedder:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")

        with (
            patch("haystack.components.embedders.watsonx_text_embedder.Embeddings") as mock_embeddings,
            patch("haystack.components.embedders.watsonx_text_embedder.Credentials") as mock_credentials,
        ):
            mock_creds_instance = MagicMock()
            mock_credentials.return_value = mock_creds_instance

            mock_embeddings_instance = MagicMock()
            mock_embeddings.return_value = mock_embeddings_instance

            yield {
                "credentials": mock_credentials,
                "embeddings": mock_embeddings,
                "creds_instance": mock_creds_instance,
                "embeddings_instance": mock_embeddings_instance,
            }

    def test_init_default(self, mock_watsonx):
        embedder = WatsonXTextEmbedder(project_id="fake-project-id")

        mock_watsonx["credentials"].assert_called_once_with(
            api_key="fake-api-key", url="https://us-south.ml.cloud.ibm.com"
        )
        mock_watsonx["embeddings"].assert_called_once_with(
            model_id="ibm/slate-30m-english-rtrvr",
            credentials=mock_watsonx["creds_instance"],
            project_id="fake-project-id",
            space_id=None,
            params=None,
        )

        assert embedder.model == "ibm/slate-30m-english-rtrvr"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self, mock_watsonx):
        embedder = WatsonXTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="ibm/slate-125m-english-rtrvr",
            url="https://custom-url.ibm.com",
            project_id="custom-project-id",
            space_id="custom-space-id",
            truncate_input_tokens=128,
            prefix="prefix ",
            suffix=" suffix",
            timeout=30.0,
            max_retries=5,
        )

        mock_watsonx["credentials"].assert_called_once_with(api_key="fake-api-key", url="https://custom-url.ibm.com")
        mock_watsonx["embeddings"].assert_called_once_with(
            model_id="ibm/slate-125m-english-rtrvr",
            credentials=mock_watsonx["creds_instance"],
            project_id="custom-project-id",
            space_id="custom-space-id",
            params={"truncate_input_tokens": 128},
        )

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            WatsonXTextEmbedder(project_id="fake-project-id")

    def test_init_fail_wo_project_or_space_id(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")
        with pytest.raises(ValueError, match="Either project_id or space_id must be provided"):
            WatsonXTextEmbedder()

    def test_to_dict(self, mock_watsonx):
        component = WatsonXTextEmbedder(project_id="fake-project-id")
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.watsonx_text_embedder.WatsonXTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-30m-english-rtrvr",
                "url": "https://us-south.ml.cloud.ibm.com",
                "project_id": "fake-project-id",
                "space_id": None,
                "truncate_input_tokens": None,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_from_dict(self, mock_watsonx):
        data = {
            "type": "haystack.components.embedders.watsonx_text_embedder.WatsonXTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-125m-english-rtrvr",
                "url": "https://custom-url.ibm.com",
                "project_id": "custom-project-id",
                "prefix": "prefix ",
                "suffix": " suffix",
            },
        }

        component = WatsonXTextEmbedder.from_dict(data)

        assert component.model == "ibm/slate-125m-english-rtrvr"
        assert component.url == "https://custom-url.ibm.com"
        assert component.project_id == "custom-project-id"
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"

    def test_prepare_input(self, mock_watsonx):
        embedder = WatsonXTextEmbedder(project_id="fake-project-id", prefix="prefix ", suffix=" suffix")
        input_text = "The food was delicious"
        prepared_input = embedder._prepare_input(input_text)
        assert prepared_input == "prefix The food was delicious suffix"

    def test_prepare_output(self, mock_watsonx):
        embedder = WatsonXTextEmbedder(
            project_id="fake-project-id", model="ibm/slate-125m-english-rtrvr", truncate_input_tokens=128
        )
        embedding = [0.1, 0.2, 0.3]
        result = embedder._prepare_output(embedding)
        assert result == {
            "embedding": [0.1, 0.2, 0.3],
            "meta": {"model": "ibm/slate-125m-english-rtrvr", "truncated_input_tokens": 128},
        }

    def test_run_wrong_input_format(self, mock_watsonx):
        embedder = WatsonXTextEmbedder(project_id="fake-project-id")
        with pytest.raises(
            TypeError,
            match="WatsonXTextEmbedder expects a string as an input. In case you want to embed a list of Documents, please use the WatsonXDocumentEmbedder.",
        ):
            embedder.run(text=[1, 2, 3])


@pytest.mark.integration
class TestWatsonXTextEmbedderIntegration:
    """Integration tests for WatsonXTextEmbedder (requires real credentials)"""

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_run(self):
        """Test real API call with simple text"""
        embedder = WatsonXTextEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            prefix="prefix ",
            suffix=" suffix",
            truncate_input_tokens=128,
        )
        result = embedder.run("The food was delicious")

        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert all(isinstance(x, float) for x in result["embedding"])
        assert "slate" in result["meta"]["model"].lower()
        assert "30m" in result["meta"]["model"].lower()

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_long_text(self):
        """Test with longer text input"""
        embedder = WatsonXTextEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=128,
        )
        long_text = "This is a test text that should work fine. " * 10
        result = embedder.run(long_text)

        assert len(result["embedding"]) > 0
        assert "model" in result["meta"]

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_different_model(self):
        """Test with a different model if available"""
        embedder = WatsonXTextEmbedder(
            model="ibm/slate-125m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=128,
        )
        result = embedder.run("Test different model")

        assert len(result["embedding"]) > 0
        assert "125m" in result["meta"]["model"]

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_text_too_long(self):
        """Test handling of text that exceeds token limit when truncation is disabled"""
        embedder = WatsonXTextEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=None,
        )

        very_long_text = "word " * 1024

        with pytest.raises(ApiRequestFailure) as exc_info:
            embedder.run(very_long_text)

        assert "exceeds the maximum sequence length" in str(exc_info.value)
        assert "512" in str(exc_info.value)

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_text_truncation(self):
        """Test that truncation works with long text"""
        embedder = WatsonXTextEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=128,
        )

        long_text = "This is a test sentence. " * 1000
        result = embedder.run(long_text)

        assert len(result["embedding"]) > 0
        assert result["meta"]["truncated_input_tokens"] == 128
