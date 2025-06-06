# SPDX-FileCopyrightText: 2023-present IBM Corporation
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from haystack.components.embedders.watsonx_document_embedder import WatsonXDocumentEmbedder
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure


class TestWatsonXDocumentEmbedder:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")

        with (
            patch("haystack.components.embedders.watsonx_document_embedder.Embeddings") as mock_embeddings,
            patch("haystack.components.embedders.watsonx_document_embedder.Credentials") as mock_credentials,
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
        embedder = WatsonXDocumentEmbedder(project_id="fake-project-id")

        mock_watsonx["credentials"].assert_called_once_with(
            api_key="fake-api-key", url="https://us-south.ml.cloud.ibm.com"
        )
        mock_watsonx["embeddings"].assert_called_once_with(
            model_id="ibm/slate-30m-english-rtrvr",
            credentials=mock_watsonx["creds_instance"],
            project_id="fake-project-id",
            space_id=None,
            params=None,
            batch_size=1000,
            concurrency_limit=5,
            max_retries=10,
        )

        assert embedder.model == "ibm/slate-30m-english-rtrvr"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 1000
        assert embedder.concurrency_limit == 5

    def test_init_with_parameters(self, mock_watsonx):
        embedder = WatsonXDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="ibm/slate-125m-english-rtrvr",
            url="https://custom-url.ibm.com",
            project_id="custom-project-id",
            space_id="custom-space-id",
            truncate_input_tokens=128,
            prefix="prefix ",
            suffix=" suffix",
            batch_size=500,
            concurrency_limit=3,
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
            batch_size=500,
            concurrency_limit=3,
            max_retries=5,
        )

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            WatsonXDocumentEmbedder(project_id="fake-project-id")

    def test_init_fail_wo_project_or_space_id(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")
        with pytest.raises(ValueError, match="Either project_id or space_id must be provided"):
            WatsonXDocumentEmbedder()

    def test_to_dict(self, mock_watsonx):
        component = WatsonXDocumentEmbedder(project_id="fake-project-id")
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.watsonx_document_embedder.WatsonXDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-30m-english-rtrvr",
                "url": "https://us-south.ml.cloud.ibm.com",
                "project_id": "fake-project-id",
                "space_id": None,
                "truncate_input_tokens": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 1000,
                "concurrency_limit": 5,
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_from_dict(self, mock_watsonx):
        data = {
            "type": "haystack.components.embedders.watsonx_document_embedder.WatsonXDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-125m-english-rtrvr",
                "url": "https://custom-url.ibm.com",
                "project_id": "custom-project-id",
                "prefix": "prefix ",
                "suffix": " suffix",
                "batch_size": 500,
                "concurrency_limit": 3,
            },
        }

        component = WatsonXDocumentEmbedder.from_dict(data)

        assert component.model == "ibm/slate-125m-english-rtrvr"
        assert component.url == "https://custom-url.ibm.com"
        assert component.project_id == "custom-project-id"
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"
        assert component.batch_size == 500
        assert component.concurrency_limit == 3

    def test_prepare_text(self, mock_watsonx):
        embedder = WatsonXDocumentEmbedder(project_id="fake-project-id", prefix="prefix ", suffix=" suffix")
        prepared_text = embedder._prepare_text("The food was delicious")
        assert prepared_text == "prefix The food was delicious suffix"

    def test_run_wrong_input_format(self, mock_watsonx):
        embedder = WatsonXDocumentEmbedder(project_id="fake-project-id")
        with pytest.raises(TypeError, match="WatsonXDocumentEmbedder expects a list of Documents as input."):
            embedder.run(documents="not a list")  # type: ignore

    def test_run_empty_documents(self, mock_watsonx):
        embedder = WatsonXDocumentEmbedder(project_id="fake-project-id")
        result = embedder.run(documents=[])
        assert result == {
            "documents": [],
            "meta": {"model": "ibm/slate-30m-english-rtrvr", "truncate_input_tokens": None, "batch_size": 1000},
        }


@pytest.mark.integration
class TestWatsonXDocumentEmbedderIntegration:
    """Integration tests for WatsonXDocumentEmbedder (requires real credentials)"""

    @pytest.fixture
    def test_documents(self):
        return [
            Document(content="The quick brown fox jumps over the lazy dog"),
            Document(content="Artificial intelligence is transforming industries"),
            Document(content="Haystack is an open-source framework for building search systems"),
        ]

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_run(self, test_documents):
        """Test real API call with documents"""
        embedder = WatsonXDocumentEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=128,
        )
        result = embedder.run(test_documents)

        assert len(result["documents"]) == 3
        for doc in result["documents"]:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

        assert result["meta"]["model"] == "ibm/slate-30m-english-rtrvr"

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_batch_processing(self, test_documents):
        """Test that batch processing works"""
        # Create enough documents to require multiple batches
        many_documents = test_documents * 50  # 150 documents

        embedder = WatsonXDocumentEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            batch_size=50,
            truncate_input_tokens=128,
        )

        result = embedder.run(many_documents)
        assert len(result["documents"]) == 150
        assert all(doc.embedding is not None for doc in result["documents"])

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_text_truncation(self, test_documents):
        """Test that truncation works with long documents"""
        # Create a document with very long content
        long_content = "This is a very long document. " * 1000
        long_document = Document(content=long_content)

        embedder = WatsonXDocumentEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            truncate_input_tokens=128,
        )

        result = embedder.run([long_document])
        assert len(result["documents"][0].embedding) > 0
        assert result["meta"]["truncate_input_tokens"] == 128

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_prefix_suffix(self, test_documents):
        """Test that prefix and suffix are correctly applied"""
        embedder = WatsonXDocumentEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            prefix="PREFIX: ",
            suffix=" :SUFFIX",
            truncate_input_tokens=128,
        )

        result = embedder.run([test_documents[0]])
        assert result["documents"][0].embedding is not None

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_concurrency_handling(self, test_documents):
        """Test that concurrency limits are respected"""
        embedder = WatsonXDocumentEmbedder(
            model="ibm/slate-30m-english-rtrvr",
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            concurrency_limit=2,
            batch_size=1,  # Force multiple batches to test concurrency
        )

        # 3 documents with batch_size=1 and concurrency_limit=2
        # should complete without errors
        result = embedder.run(test_documents)
        assert len(result["documents"]) == 3
