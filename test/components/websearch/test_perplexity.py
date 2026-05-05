# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from httpx import ConnectTimeout, HTTPStatusError, Request, RequestError, Response

from haystack import Document
from haystack.components.websearch.perplexity import PerplexityError, PerplexityWebSearch
from haystack.utils.auth import Secret

EXAMPLE_PERPLEXITY_RESPONSE = {
    "results": [
        {
            "title": "Satya Nadella - Stories",
            "url": "https://news.microsoft.com/exec/satya-nadella/",
            "snippet": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
            "date": "2023-11-22",
            "last_updated": "2023-11-22",
        },
        {
            "title": "Satya Nadella - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Satya_Nadella",
            "snippet": "Satya Narayana Nadella is an Indian-American business executive.",
            "date": "2023-11-21",
            "last_updated": "2023-11-22",
        },
        {
            "title": "Satya Nadella | Forbes Profile",
            "url": "https://www.forbes.com/profile/satya-nadella/",
            "snippet": "Satya Nadella replaced billionaire Steve Ballmer as Microsoft CEO in 2014.",
            "date": "2023-11-20",
            "last_updated": "2023-11-22",
        },
        {
            "title": "Microsoft CEO Satya Nadella",
            "url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
            "snippet": "Microsoft CEO Satya Nadella speaks during the OpenAI DevDay event.",
            "date": "2023-11-19",
            "last_updated": "2023-11-22",
        },
        {
            "title": "Satya Nadella's response to the OpenAI board",
            "url": "https://fortune.com/2023/11/21/microsoft-ceo-satya-nadella/",
            "snippet": "Microsoft CEO Satya Nadella's response to the OpenAI board changes.",
            "date": "2023-11-21",
            "last_updated": "2023-11-22",
        },
        {
            "title": "5 Facts About Microsoft CEO Satya Nadella",
            "url": "https://in.benzinga.com/content/5-facts-microsoft-ceo-satya-nadella",
            "snippet": "Satya Nadella's journey at Microsoft underscores diverse experiences.",
            "date": "2023-11-22",
            "last_updated": "2023-11-22",
        },
        {
            "title": "Satya Nadella @satyanadella on X",
            "url": "https://twitter.com/satyanadella",
            "snippet": "Chairman and CEO of Microsoft Corporation.",
            "date": "2023-11-22",
            "last_updated": "2023-11-22",
        },
    ],
    "id": "search_test_id",
    "server_time": "2023-11-22T16:10:56Z",
}


@pytest.fixture
def mock_perplexity_search_result() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.perplexity.httpx.post") as mock_post:
        mock_post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_PERPLEXITY_RESPONSE)
        yield mock_post


@pytest.fixture
def mock_perplexity_search_result_async() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.perplexity.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_PERPLEXITY_RESPONSE)
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        yield mock_client_cls


class TestPerplexityWebSearch:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            PerplexityWebSearch()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")
        component = PerplexityWebSearch(top_k=10, search_params={"country": "US"})
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.websearch.perplexity.PerplexityWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["PERPLEXITY_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 10,
                "search_params": {"country": "US"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "test-api-key")
        data = {
            "type": "haystack.components.websearch.perplexity.PerplexityWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["PERPLEXITY_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 5,
                "search_params": {"country": "US"},
            },
        }
        component = PerplexityWebSearch.from_dict(data)
        assert component.top_k == 5
        assert component.search_params == {"country": "US"}

    @pytest.mark.parametrize("top_k", [1, 5, 7])
    def test_web_search_top_k(self, mock_perplexity_search_result: MagicMock, top_k: int) -> None:
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    def test_attribution_header_sent(self, mock_perplexity_search_result: MagicMock) -> None:
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"), top_k=3)
        ws.run(query="test query")
        _, kwargs = mock_perplexity_search_result.call_args
        headers = kwargs["headers"]
        assert "X-Pplx-Integration" in headers
        assert headers["X-Pplx-Integration"].startswith("haystack-core/")
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_search_params_in_payload(self, mock_perplexity_search_result: MagicMock) -> None:
        ws = PerplexityWebSearch(
            api_key=Secret.from_token("test-api-key"),
            top_k=5,
            search_params={"country": "US", "search_recency_filter": "week"},
        )
        ws.run(query="test query")
        _, kwargs = mock_perplexity_search_result.call_args
        payload = kwargs["json"]
        assert payload["query"] == "test query"
        assert payload["max_results"] == 5
        assert payload["country"] == "US"
        assert payload["search_recency_filter"] == "week"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 5, 7])
    async def test_web_search_top_k_async(self, mock_perplexity_search_result_async: MagicMock, top_k: int) -> None:
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = await ws.run_async(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @patch("httpx.post")
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_timeout_error_async(self, mock_post: AsyncMock) -> None:
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @patch("httpx.post")
    def test_request_exception(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(PerplexityError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_request_exception_async(self, mock_post: AsyncMock) -> None:
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(PerplexityError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @patch("httpx.post")
    def test_bad_response_code(self, mock_post: MagicMock) -> None:
        mock_response = mock_post.return_value
        mock_response.status_code = 404
        mock_error_request = Request("POST", "https://example.com")
        mock_error_response = Response(404)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=mock_error_request, response=mock_error_response
        )
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(PerplexityError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_bad_response_code_async(self, mock_client_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_response = Mock(status_code=404)
        mock_error_request = Request("POST", "https://example.com")
        mock_error_response = Response(404)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=mock_error_request, response=mock_error_response
        )
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        ws = PerplexityWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(PerplexityError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @pytest.mark.skipif(
        not os.environ.get("PERPLEXITY_API_KEY", None),
        reason="Export an env var called PERPLEXITY_API_KEY containing the Perplexity API key to run this test.",
    )
    @pytest.mark.integration
    def test_web_search(self) -> None:
        ws = PerplexityWebSearch(top_k=10)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("PERPLEXITY_API_KEY", None),
        reason="Export an env var called PERPLEXITY_API_KEY containing the Perplexity API key to run this test.",
    )
    @pytest.mark.integration
    async def test_web_search_async(self) -> None:
        ws = PerplexityWebSearch(top_k=10)
        results = await ws.run_async(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)
