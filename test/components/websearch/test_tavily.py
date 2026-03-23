# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from httpx import ConnectTimeout, HTTPStatusError, Request, RequestError, Response

from haystack import Document
from haystack.components.websearch.tavily import TavilyError, TavilyWebSearch
from haystack.utils.auth import Secret

EXAMPLE_TAVILY_RESPONSE = {
    "query": "Who is CEO of Microsoft?",
    "follow_up_questions": None,
    "answer": None,
    "images": [],
    "results": [
        {
            "title": "Satya Nadella - Stories",
            "url": "https://news.microsoft.com/exec/satya-nadella/",
            "content": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft. "
            "Before being named CEO in February 2014, Nadella held leadership roles in both ...",
            "score": 0.98,
            "raw_content": None,
        },
        {
            "title": "Satya Nadella",
            "url": "https://en.wikipedia.org/wiki/Satya_Nadella",
            "content": "Satya Narayana Nadella is an Indian-American business executive. "
            "He is the executive chairman and CEO of Microsoft.",
            "score": 0.95,
            "raw_content": None,
        },
        {
            "title": "Satya Nadella",
            "url": "https://www.linkedin.com/in/satyanadella",
            "content": "As chairman and CEO of Microsoft, I define my mission and that of "
            "my company as empowering every person and every organization on the planet to achieve more.",
            "score": 0.92,
            "raw_content": None,
        },
        {
            "title": "Who is Satya Nadella",
            "url": "https://www.business-standard.com/about/who-is-satya-nadella",
            "content": "Satya Narayana Nadella is the chief executive officer (CEO) of Microsoft.",
            "score": 0.90,
            "raw_content": None,
        },
        {
            "title": "Satya Nadella (@satyanadella) / X",
            "url": "https://twitter.com/satyanadella",
            "content": "Chairman and CEO of Microsoft Corporation.",
            "score": 0.85,
            "raw_content": None,
        },
        {
            "title": "Satya Nadella | Biography & Facts",
            "url": "https://www.britannica.com/biography/Satya-Nadella",
            "content": "Satya Nadella (born August 19, 1967, Hyderabad, India) Indian-born business executive.",
            "score": 0.83,
            "raw_content": None,
        },
        {
            "title": "Satya Nadella",
            "url": "https://www.forbes.com/profile/satya-nadella/",
            "content": "Satya Nadella replaced billionaire Steve Ballmer as Microsoft CEO in 2014.",
            "score": 0.80,
            "raw_content": None,
        },
    ],
    "response_time": 1.5,
}


@pytest.fixture
def mock_tavily_search_result() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.tavily.httpx.post") as mock_post:
        mock_post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_TAVILY_RESPONSE)
        yield mock_post


@pytest.fixture
def mock_tavily_search_result_async() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.tavily.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_TAVILY_RESPONSE)
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        yield mock_client_cls


class TestTavilyWebSearch:
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            TavilyWebSearch()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-api-key")
        component = TavilyWebSearch(top_k=10, search_params={"search_depth": "advanced"})
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.websearch.tavily.TavilyWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 10,
                "search_params": {"search_depth": "advanced"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-api-key")
        data = {
            "type": "haystack.components.websearch.tavily.TavilyWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["TAVILY_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 5,
                "search_params": {"topic": "news"},
            },
        }
        component = TavilyWebSearch.from_dict(data)
        assert component.top_k == 5
        assert component.search_params == {"topic": "news"}

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_web_search_top_k(self, mock_tavily_search_result: MagicMock, top_k: int) -> None:
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 3, 5])
    async def test_web_search_top_k_async(self, mock_tavily_search_result_async: MagicMock, top_k: int) -> None:
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
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
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_timeout_error_async(self, mock_client_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectTimeout("Request has timed out.")
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @patch("httpx.post")
    def test_request_exception(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TavilyError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_request_exception_async(self, mock_client_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.post.side_effect = RequestError("An error has occurred in the request.")
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TavilyError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @patch("httpx.post")
    def test_bad_response_code(self, mock_post: MagicMock) -> None:
        mock_response = mock_post.return_value
        mock_response.status_code = 401
        mock_error_request = Request("POST", "https://api.tavily.com/search")
        mock_error_response = Response(401)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "401 Unauthorized.", request=mock_error_request, response=mock_error_response
        )
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TavilyError):
            ws.run(query="Who is CEO of Microsoft?")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_bad_response_code_async(self, mock_client_cls: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_response = Mock(status_code=401)
        mock_error_request = Request("POST", "https://api.tavily.com/search")
        mock_error_response = Response(401)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "401 Unauthorized.", request=mock_error_request, response=mock_error_response
        )
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client_cls.return_value = mock_client
        ws = TavilyWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TavilyError):
            await ws.run_async(query="Who is CEO of Microsoft?")

    @pytest.mark.skipif(
        not os.environ.get("TAVILY_API_KEY", None),
        reason="Export an env var called TAVILY_API_KEY containing the Tavily API key to run this test.",
    )
    @pytest.mark.integration
    def test_web_search(self) -> None:
        ws = TavilyWebSearch(top_k=5)
        results = ws.run(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 5
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("TAVILY_API_KEY", None),
        reason="Export an env var called TAVILY_API_KEY containing the Tavily API key to run this test.",
    )
    @pytest.mark.integration
    async def test_web_search_async(self) -> None:
        ws = TavilyWebSearch(top_k=5)
        results = await ws.run_async(query="Who is CEO of Microsoft?")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 5
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)
