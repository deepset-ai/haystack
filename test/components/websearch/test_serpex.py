# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import httpx
import pytest

from haystack.components.websearch.serpex import SerpexWebSearch
from haystack.dataclasses import Document
from haystack.utils import Secret


@pytest.fixture
def mock_serpex_response():
    """Mock SERPEX API response with results"""
    return {
        "results": [
            {
                "title": "Haystack - Open Source LLM Framework",
                "url": "https://haystack.deepset.ai/",
                "snippet": "Haystack is an open-source framework for building production-ready LLM applications.",
                "position": 1,
            },
            {
                "title": "Haystack Documentation",
                "url": "https://docs.haystack.deepset.ai/",
                "snippet": "Complete documentation for Haystack framework.",
                "position": 2,
            },
            {
                "title": "Haystack GitHub Repository",
                "url": "https://github.com/deepset-ai/haystack",
                "snippet": "Official Haystack GitHub repository with source code.",
                "position": 3,
            },
        ]
    }


@pytest.fixture
def mock_empty_serpex_response():
    """Mock SERPEX API response with no results"""
    return {"results": []}


class TestSerpexWebSearch:
    def test_init_default_params(self):
        """Test initialization with default parameters"""
        fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
        assert fetcher.api_key.resolve_value() == "test-api-key"
        assert fetcher.engine == "google"
        assert fetcher.num_results == 10
        assert fetcher.timeout == 10.0
        assert fetcher.retry_attempts == 2
        assert hasattr(fetcher, "_client")
        assert isinstance(fetcher._client, httpx.Client)

    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        fetcher = SerpexWebSearch(
            api_key=Secret.from_token("test-key"),
            engine="bing",
            num_results=5,
            timeout=20.0,
            retry_attempts=3,
        )
        assert fetcher.api_key.resolve_value() == "test-key"
        assert fetcher.engine == "bing"
        assert fetcher.num_results == 5
        assert fetcher.timeout == 20.0
        assert fetcher.retry_attempts == 3

    def test_to_dict(self):
        """Test serialization to dictionary"""
        fetcher = SerpexWebSearch(
            api_key=Secret.from_token("test-api-key"),
            engine="duckduckgo",
            num_results=15,
        )
        data = fetcher.to_dict()
        assert data["init_parameters"]["api_key"]["type"] == "env_var"
        assert data["init_parameters"]["engine"] == "duckduckgo"
        assert data["init_parameters"]["num_results"] == 15
        assert data["init_parameters"]["timeout"] == 10.0
        assert data["init_parameters"]["retry_attempts"] == 2

    def test_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "type": "haystack.components.websearch.serpex.SerpexWebSearch",
            "init_parameters": {
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["SERPEX_API_KEY"],
                    "strict": True,
                },
                "engine": "brave",
                "num_results": 20,
                "timeout": 15.0,
                "retry_attempts": 1,
            },
        }
        fetcher = SerpexWebSearch.from_dict(data)
        assert fetcher.engine == "brave"
        assert fetcher.num_results == 20
        assert fetcher.timeout == 15.0
        assert fetcher.retry_attempts == 1

    def test_run_with_mock_response(self, mock_serpex_response):
        """Test run method with mocked successful API response"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
            result = fetcher.run(query="What is Haystack?")

            documents = result["documents"]
            assert len(documents) == 3

            # Check first document
            assert isinstance(documents[0], Document)
            assert (
                documents[0].content
                == "Haystack is an open-source framework for building production-ready LLM applications."
            )
            assert documents[0].meta["title"] == "Haystack - Open Source LLM Framework"
            assert documents[0].meta["url"] == "https://haystack.deepset.ai/"
            assert documents[0].meta["position"] == 1
            assert documents[0].meta["query"] == "What is Haystack?"
            assert documents[0].meta["engine"] == "google"

            # Check second document
            assert documents[1].meta["title"] == "Haystack Documentation"
            assert documents[1].meta["position"] == 2

            # Check third document
            assert documents[2].meta["title"] == "Haystack GitHub Repository"
            assert documents[2].meta["position"] == 3

    def test_run_with_empty_results(self, mock_empty_serpex_response):
        """Test run method with empty results"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_empty_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
            result = fetcher.run(query="nonexistent query")

            documents = result["documents"]
            assert len(documents) == 0

    def test_run_with_engine_override(self, mock_serpex_response):
        """Test run method with engine parameter override"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(
                api_key=Secret.from_token("test-api-key"), engine="google"
            )
            result = fetcher.run(query="test query", engine="bing")

            # Verify the request was made with the overridden engine
            call_args = mock_get.call_args
            assert call_args[1]["params"]["engine"] == "bing"

            documents = result["documents"]
            assert len(documents) == 3
            assert documents[0].meta["engine"] == "bing"

    def test_run_with_num_results_override(self, mock_serpex_response):
        """Test run method with num_results parameter override"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(
                api_key=Secret.from_token("test-api-key"), num_results=10
            )
            fetcher.run(query="test query", num_results=5)

            # Verify the request was made with the overridden num_results
            call_args = mock_get.call_args
            assert call_args[1]["params"]["num"] == 5

    def test_run_with_time_range(self, mock_serpex_response):
        """Test run method with time_range parameter"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
            fetcher.run(query="test query", time_range="week")

            # Verify the request was made with time_range parameter
            call_args = mock_get.call_args
            assert call_args[1]["params"]["time_range"] == "week"

    def test_run_with_http_error(self):
        """Test run method with HTTP error"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=401)
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "401 Unauthorized", request=Mock(), response=mock_response
            )
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("invalid-key"))
            with pytest.raises(httpx.HTTPStatusError):
                fetcher.run(query="test query")

    def test_run_with_network_error(self):
        """Test run method with network error"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_get.side_effect = httpx.RequestError(
                "Connection failed", request=Mock()
            )

            fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
            with pytest.raises(httpx.RequestError):
                fetcher.run(query="test query")

    def test_run_verifies_api_key_in_headers(self, mock_serpex_response):
        """Test that API key is properly included in request headers"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("secret-key-123"))
            fetcher.run(query="test query")

            # Verify the Authorization header was set correctly
            call_args = mock_get.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer secret-key-123"
            assert call_args[1]["headers"]["Content-Type"] == "application/json"

    def test_run_uses_correct_api_endpoint(self, mock_serpex_response):
        """Test that the correct SERPEX API endpoint is used"""
        with patch("haystack.components.websearch.serpex.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200)
            mock_response.json.return_value = mock_serpex_response
            mock_get.return_value = mock_response

            fetcher = SerpexWebSearch(api_key=Secret.from_token("test-api-key"))
            fetcher.run(query="test query")

            # Verify the correct endpoint was called
            call_args = mock_get.call_args
            assert call_args[0][0] == "https://api.serpex.dev/api/search"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("SERPEX_API_KEY"), reason="SERPEX_API_KEY not set"
    )
    def test_run_with_real_api(self):
        """Integration test with real SERPEX API"""
        api_key = os.environ.get("SERPEX_API_KEY")
        fetcher = SerpexWebSearch(api_key=api_key)

        result = fetcher.run(query="Haystack LLM framework")
        documents = result["documents"]

        # Basic assertions
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all("title" in doc.meta for doc in documents)
        assert all("url" in doc.meta for doc in documents)
        assert all(doc.content for doc in documents)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("SERPEX_API_KEY"), reason="SERPEX_API_KEY not set"
    )
    def test_run_with_different_engines(self):
        """Integration test with different search engines"""
        api_key = os.environ.get("SERPEX_API_KEY")

        # Test Google
        fetcher_google = SerpexWebSearch(api_key=api_key, engine="google")
        result_google = fetcher_google.run(query="Python programming")
        assert len(result_google["documents"]) > 0

        # Test DuckDuckGo
        fetcher_ddg = SerpexWebSearch(api_key=api_key, engine="duckduckgo")
        result_ddg = fetcher_ddg.run(query="Python programming")
        assert len(result_ddg["documents"]) > 0
