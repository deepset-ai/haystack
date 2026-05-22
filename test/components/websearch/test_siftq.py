# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from httpx import ConnectTimeout, HTTPStatusError, Request, RequestError, Response

from haystack import Document
from haystack.components.websearch.siftq import SIFTQ_DEFAULT_API_KEY, SiftqError, SiftqWebSearch
from haystack.utils.auth import Secret

EXAMPLE_SIFTQ_RESPONSE = {
    "credits": 1,
    "total": 3,
    "webpages": [
        {
            "title": "Paris - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Paris",
            "snippet": "Paris is the capital and largest city of France.",
            "score": "0.95",
            "position": 0,
        },
        {
            "title": "Paris travel guide",
            "link": "https://example.com/paris-guide",
            "snippet": "The best things to do in Paris, France.",
            "score": "0.85",
            "position": 1,
        },
        {
            "title": "France info",
            "link": "https://example.com/france",
            "snippet": "General information about France and its capital Paris.",
            "score": "0.75",
            "position": 2,
        },
    ],
}


@pytest.fixture
def mock_siftq_search_result() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.siftq.httpx") as mock_run:
        mock_run.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_SIFTQ_RESPONSE)
        yield mock_run


@pytest.fixture
def mock_siftq_search_result_async() -> Generator[MagicMock, None, None]:
    with patch("haystack.components.websearch.siftq.httpx.AsyncClient") as mock_run:
        mock_client = AsyncMock()
        mock_client.post.return_value = Mock(status_code=200, json=lambda: EXAMPLE_SIFTQ_RESPONSE)
        mock_client.__aenter__.return_value = mock_client
        mock_run.return_value = mock_client
        yield mock_run


class TestSiftqWebSearch:
    def test_init_default_params(self):
        ws = SiftqWebSearch()
        assert ws.top_k == 10
        assert ws.scope == "webpage"
        assert ws.allowed_domains is None
        assert ws.search_params == {}

    def test_init_custom_params(self):
        ws = SiftqWebSearch(
            api_key=Secret.from_token("test-key"),
            top_k=5,
            scope="scholar",
            allowed_domains=["example.com"],
            search_params={"includeSummary": True},
        )
        assert ws.top_k == 5
        assert ws.scope == "scholar"
        assert ws.allowed_domains == ["example.com"]
        assert ws.search_params == {"includeSummary": True}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("SIFTQ_API_KEY", "test-api-key")
        ws = SiftqWebSearch(top_k=10, scope="webpage")
        data = ws.to_dict()
        assert data == {
            "type": "haystack.components.websearch.siftq.SiftqWebSearch",
            "init_parameters": {
                "api_key": {"env_vars": ["SIFTQ_API_KEY"], "strict": False, "type": "env_var"},
                "top_k": 10,
                "scope": "webpage",
                "allowed_domains": None,
                "search_params": {},
            },
        }

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_web_search_top_k(self, mock_siftq_search_result: MagicMock, top_k: int) -> None:
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = ws.run(query="capital of France")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 2, 3])
    async def test_web_search_top_k_async(self, mock_siftq_search_result_async: MagicMock, top_k: int) -> None:
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"), top_k=top_k)
        results = await ws.run_async(query="capital of France")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == top_k
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @patch("haystack.components.websearch.siftq.httpx.post")
    def test_prepare_request_with_allowed_domains(self, mock_post: MagicMock) -> None:
        mock_post.return_value = Mock(status_code=200, json=lambda: {"webpages": []})
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"), allowed_domains=["example.com", "test.org"])
        ws.run(query="test query")
        call_kwargs = mock_post.call_args[1]
        assert "site:example.com" in call_kwargs["json"]["q"]
        assert "site:test.org" in call_kwargs["json"]["q"]
        assert call_kwargs["json"]["scope"] == "webpage"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="Unknown scope"):
            SiftqWebSearch(scope="invalid")

    def test_document_metadata(self, mock_siftq_search_result: MagicMock) -> None:
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"), top_k=1)
        results = ws.run(query="capital of France")
        doc = results["documents"][0]
        assert doc.meta["title"] == "Paris - Wikipedia"
        assert doc.meta["link"] == "https://en.wikipedia.org/wiki/Paris"
        assert doc.meta["score"] == "0.95"
        assert doc.meta["position"] == 0

    @patch("httpx.post")
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            ws.run(query="capital of France")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_timeout_error_async(self, mock_post: AsyncMock) -> None:
        mock_post.side_effect = ConnectTimeout("Request has timed out.")
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TimeoutError):
            await ws.run_async(query="capital of France")

    @patch("httpx.post")
    def test_request_exception(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SiftqError):
            ws.run(query="capital of France")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_request_exception_async(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RequestError("An error has occurred in the request.")
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SiftqError):
            await ws.run_async(query="capital of France")

    @patch("httpx.post")
    def test_bad_response_code(self, mock_post: MagicMock) -> None:
        mock_response = mock_post.return_value
        mock_response.status_code = 404
        mock_error_request = Request("POST", "https://example.com")
        mock_error_response = Response(404)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=mock_error_request, response=mock_error_response
        )
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SiftqError):
            ws.run(query="capital of France")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_bad_response_code_async(self, mock_run: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_response = Mock(status_code=404)
        mock_error_request = Request("POST", "https://example.com")
        mock_error_response = Response(404)
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "404 Not Found.", request=mock_error_request, response=mock_error_response
        )
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_run.return_value = mock_client
        ws = SiftqWebSearch(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(SiftqError):
            await ws.run_async(query="capital of France")

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SIFTQ_API_KEY", "env-key-value")
        ws = SiftqWebSearch()
        assert ws._resolve_api_key() == "env-key-value"

    def test_resolve_api_key_default_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("SIFTQ_API_KEY", raising=False)
        ws = SiftqWebSearch()
        key = ws._resolve_api_key()
        assert key == SIFTQ_DEFAULT_API_KEY

    def test_resolve_api_key_token_takes_precedence(self):
        ws = SiftqWebSearch(api_key=Secret.from_token("explicit-key"))
        assert ws._resolve_api_key() == "explicit-key"

    def test_api_error_2005_key_rejected(self):
        error_body = {"code": 2005, "message": "API key rejected"}
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack.components.websearch.siftq.httpx.post") as mock_post:
            response = Mock(status_code=200)
            response.json.return_value = error_body
            mock_post.return_value = response

            with pytest.raises(SiftqError, match="API key rejected"):
                ws.run(query="test")

    def test_api_error_3003_limit_reached(self):
        error_body = {"code": 3003, "message": "daily limit reached"}
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack.components.websearch.siftq.httpx.post") as mock_post:
            response = Mock(status_code=200)
            response.json.return_value = error_body
            mock_post.return_value = response

            with pytest.raises(SiftqError, match="daily search limit reached"):
                ws.run(query="test")

    def test_api_error_unknown_code(self):
        error_body = {"code": 9999, "message": "something went wrong"}
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack.components.websearch.siftq.httpx.post") as mock_post:
            response = Mock(status_code=200)
            response.json.return_value = error_body
            mock_post.return_value = response

            with pytest.raises(SiftqError, match="something went wrong"):
                ws.run(query="test")

    def test_api_error_no_code_passes_through(self):
        normal_body = {"webpages": [{"title": "ok", "link": "https://example.com", "snippet": "test"}]}
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack.components.websearch.siftq.httpx.post") as mock_post:
            response = Mock(status_code=200)
            response.json.return_value = normal_body
            mock_post.return_value = response

            result = ws.run(query="test")
            assert len(result["documents"]) > 0

    @pytest.mark.asyncio
    async def test_api_error_2005_async(self):
        error_body = {"code": 2005, "message": "API key rejected"}
        ws = SiftqWebSearch(api_key=Secret.from_token("test-key"))

        with patch("haystack.components.websearch.siftq.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            response = Mock(status_code=200)
            response.json.return_value = error_body
            mock_client.post.return_value = response
            mock_client.__aenter__.return_value = mock_client
            mock_client_cls.return_value = mock_client

            with pytest.raises(SiftqError, match="API key rejected"):
                await ws.run_async(query="test")

    @pytest.mark.skipif(
        not os.environ.get("SIFTQ_API_KEY", None),
        reason="Export an env var called SIFTQ_API_KEY containing the SiftQ API key to run this test.",
    )
    @pytest.mark.integration
    def test_web_search(self) -> None:
        ws = SiftqWebSearch(top_k=10)
        results = ws.run(query="capital of France")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("SIFTQ_API_KEY", None),
        reason="Export an env var called SIFTQ_API_KEY containing the SiftQ API key to run this test.",
    )
    @pytest.mark.integration
    async def test_web_search_async(self) -> None:
        ws = SiftqWebSearch(top_k=10)
        results = await ws.run_async(query="capital of France")
        documents = results["documents"]
        links = results["links"]
        assert len(documents) == len(links) == 10
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(isinstance(link, str) for link in links)
        assert all(link.startswith("http") for link in links)