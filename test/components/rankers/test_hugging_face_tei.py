# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch, MagicMock

import requests
import httpx

from haystack import Document
from haystack.components.rankers.hugging_face_tei import HuggingFaceTEIRanker, TruncationDirection
from haystack.utils import Secret


class TestHuggingFaceTEIRanker:
    def test_init(self, monkeypatch):
        """Test initialization with default and custom parameters"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Default parameters
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        assert ranker.url == "https://api.my-tei-service.com"
        assert ranker.top_k == 10
        assert ranker.timeout == 30
        assert not ranker.token.resolve_value()
        assert ranker.max_retries == 3
        assert ranker.retry_status_codes is None

        # Custom parameters
        token = Secret.from_token("my_api_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=5,
            timeout=60,
            token=token,
            max_retries=5,
            retry_status_codes=[500, 502, 503],
        )
        assert ranker.url == "https://api.my-tei-service.com"
        assert ranker.top_k == 5
        assert ranker.timeout == 60
        assert ranker.token == token
        assert ranker.max_retries == 5
        assert ranker.retry_status_codes == [500, 502, 503]

    def test_to_dict(self, monkeypatch):
        """Test serialization to dict with Secret token"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        component = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com", top_k=5, timeout=30, max_retries=4, retry_status_codes=[500, 502]
        )
        data = component.to_dict()

        assert data["type"] == "haystack.components.rankers.hugging_face_tei.HuggingFaceTEIRanker"
        assert data["init_parameters"]["url"] == "https://api.my-tei-service.com"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["timeout"] == 30
        assert data["init_parameters"]["token"] == {
            "env_vars": ["HF_API_TOKEN", "HF_TOKEN"],
            "strict": False,
            "type": "env_var",
        }
        assert data["init_parameters"]["max_retries"] == 4
        assert data["init_parameters"]["retry_status_codes"] == [500, 502]

    def test_from_dict(self, monkeypatch):
        """Test deserialization from dict with environment variable token"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        data = {
            "type": "haystack.components.rankers.hugging_face_tei.HuggingFaceTEIRanker",
            "init_parameters": {
                "url": "https://api.my-tei-service.com",
                "top_k": 5,
                "timeout": 30,
                "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                "max_retries": 4,
                "retry_status_codes": [500, 502],
            },
        }

        component = HuggingFaceTEIRanker.from_dict(data)

        assert component.url == "https://api.my-tei-service.com"
        assert component.top_k == 5
        assert component.timeout == 30
        assert component.max_retries == 4
        assert component.retry_status_codes == [500, 502]

    def test_empty_documents(self, monkeypatch):
        """Test that empty documents list returns empty result"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        result = ranker.run(query="test query", documents=[])
        assert result == {"documents": []}

    @patch("haystack.components.rankers.hugging_face_tei.request_with_retry")
    def test_run_with_mock(self, mock_request, monkeypatch):
        """Test run method with mocked API response"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Setup mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.return_value = [
            {"index": 2, "score": 0.95},
            {"index": 1, "score": 0.85},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=3,
            timeout=30,
            token=token,
            max_retries=4,
            retry_status_codes=[500, 502],
        )

        docs = [Document(content="Document A"), Document(content="Document B"), Document(content="Document C")]

        # Run the ranker
        result = ranker.run(query="test query", documents=docs)

        # Check that request_with_retry was called with correct parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["Document A", "Document B", "Document C"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=4,
            status_codes_to_retry=[500, 502],
        )

        # Check that documents are ranked correctly
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document C"
        assert result["documents"][0].score == 0.95
        assert result["documents"][1].content == "Document B"
        assert result["documents"][1].score == 0.85
        assert result["documents"][2].content == "Document A"
        assert result["documents"][2].score == 0.75

    @patch("haystack.components.rankers.hugging_face_tei.request_with_retry")
    def test_run_with_truncation_direction(self, mock_request, monkeypatch):
        """Test run method with truncation direction parameter"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Setup mock response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.return_value = [{"index": 0, "score": 0.95}]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com", token=token)
        docs = [Document(content="Document A")]

        # Run the ranker with truncation direction
        ranker.run(query="test query", documents=docs, truncation_direction=TruncationDirection.LEFT)

        # Check that request includes truncation parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={
                "query": "test query",
                "texts": ["Document A"],
                "raw_scores": False,
                "truncate": True,
                "truncation_direction": "Left",
            },
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=3,
            status_codes_to_retry=None,
        )

    @patch("haystack.components.rankers.hugging_face_tei.request_with_retry")
    def test_run_with_custom_top_k(self, mock_request, monkeypatch):
        """Test run method with custom top_k parameter"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Setup mock response with 5 documents
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.return_value = [
            {"index": 4, "score": 0.95},
            {"index": 3, "score": 0.90},
            {"index": 2, "score": 0.85},
            {"index": 1, "score": 0.80},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker with top_k=3
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com", top_k=3)

        # Create 5 test documents
        docs = [Document(content=f"Document {i}") for i in range(5)]

        # Run the ranker
        result = ranker.run(query="test query", documents=docs)

        # Check that only top 3 documents are returned
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document 4"
        assert result["documents"][1].content == "Document 3"
        assert result["documents"][2].content == "Document 2"

        # Test with run-time top_k override
        result = ranker.run(query="test query", documents=docs, top_k=2)

        # Check that only top 2 documents are returned
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Document 4"
        assert result["documents"][1].content == "Document 3"

    @patch("haystack.components.rankers.hugging_face_tei.request_with_retry")
    def test_error_handling(self, mock_request, monkeypatch):
        """Test error handling in the ranker"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Setup mock response with error
        mock_response = MagicMock(spec=requests.Response)
        mock_response.json.return_value = {"error": "Some error occurred", "error_type": "TestError"}
        mock_request.return_value = mock_response

        # Create ranker and test documents
        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        docs = [Document(content="Document A")]

        # Test that RuntimeError is raised with the correct message
        with pytest.raises(
            RuntimeError, match=r"HuggingFaceTEIRanker API call failed \(TestError\): Some error occurred"
        ):
            ranker.run(query="test query", documents=docs)

        # Test unexpected response format
        mock_response.json.return_value = {"unexpected": "format"}
        with pytest.raises(RuntimeError, match="Unexpected response format from text-embeddings-inference rerank API"):
            ranker.run(query="test query", documents=docs)

    @pytest.mark.asyncio
    @patch("haystack.components.rankers.hugging_face_tei.async_request_with_retry")
    async def test_run_async_with_mock(self, mock_request, monkeypatch):
        """Test run_async method with mocked API response"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Setup mock response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = [
            {"index": 2, "score": 0.95},
            {"index": 1, "score": 0.85},
            {"index": 0, "score": 0.75},
        ]
        mock_request.return_value = mock_response

        # Create ranker and test documents
        token = Secret.from_token("test_token")
        ranker = HuggingFaceTEIRanker(
            url="https://api.my-tei-service.com",
            top_k=3,
            timeout=30,
            token=token,
            max_retries=4,
            retry_status_codes=[500, 502],
        )

        docs = [Document(content="Document A"), Document(content="Document B"), Document(content="Document C")]

        # Run the ranker asynchronously
        result = await ranker.run_async(query="test query", documents=docs)

        # Check that async_request_with_retry was called with correct parameters
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.my-tei-service.com/rerank",
            json={"query": "test query", "texts": ["Document A", "Document B", "Document C"], "raw_scores": False},
            timeout=30,
            headers={"Authorization": "Bearer test_token"},
            attempts=4,
            status_codes_to_retry=[500, 502],
        )

        # Check that documents are ranked correctly
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Document C"
        assert result["documents"][0].score == 0.95
        assert result["documents"][1].content == "Document B"
        assert result["documents"][1].score == 0.85
        assert result["documents"][2].content == "Document A"
        assert result["documents"][2].score == 0.75

    @pytest.mark.asyncio
    @patch("haystack.components.rankers.hugging_face_tei.async_request_with_retry")
    async def test_run_async_empty_documents(self, mock_request, monkeypatch):
        """Test run_async with empty documents list"""
        # Ensure we're not using system environment variables
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        ranker = HuggingFaceTEIRanker(url="https://api.my-tei-service.com")
        result = await ranker.run_async(query="test query", documents=[])

        # Check that no API call was made
        mock_request.assert_not_called()
        assert result == {"documents": []}
