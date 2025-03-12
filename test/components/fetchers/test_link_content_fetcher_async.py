# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch, Mock

import pytest
import httpx

from haystack.components.fetchers.link_content import LinkContentFetcher, DEFAULT_USER_AGENT

HTML_URL = "https://docs.haystack.deepset.ai/docs"
TEXT_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
PDF_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/b5987a6d8d0714eb2f3011183ab40093d2e4a41a/e2e/samples/pipelines/sample_pdf_1.pdf"


class TestLinkContentFetcherAsync:
    @pytest.mark.asyncio
    async def test_run_async(self):
        """Test basic async fetching with a mocked response"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock(status_code=200, text="Example test response", headers={"Content-Type": "text/plain"})
            mock_get.return_value = mock_response

            fetcher = LinkContentFetcher()
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]

            first_stream = streams[0]
            assert first_stream.data == b"Example test response"
            assert first_stream.meta["content_type"] == "text/plain"
            assert first_stream.mime_type == "text/plain"

    @pytest.mark.asyncio
    async def test_run_async_multiple(self):
        """Test async fetching of multiple URLs with mocked responses"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock(status_code=200, text="Example test response", headers={"Content-Type": "text/plain"})
            mock_get.return_value = mock_response

            fetcher = LinkContentFetcher()
            streams = (await fetcher.run_async(urls=["https://www.example1.com", "https://www.example2.com"]))[
                "streams"
            ]

            assert len(streams) == 2
            for stream in streams:
                assert stream.data == b"Example test response"
                assert stream.meta["content_type"] == "text/plain"
                assert stream.mime_type == "text/plain"

    @pytest.mark.asyncio
    async def test_run_async_empty_urls(self):
        """Test async fetching with empty URL list"""
        fetcher = LinkContentFetcher()
        streams = (await fetcher.run_async(urls=[]))["streams"]
        assert len(streams) == 0

    @pytest.mark.asyncio
    async def test_run_async_error_handling(self):
        """Test error handling for async fetching"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.get") as mock_get:
            mock_response = Mock(status_code=404)
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
            mock_get.return_value = mock_response

            # With raise_on_failure=False
            fetcher = LinkContentFetcher(raise_on_failure=False)
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]
            assert len(streams) == 1  # Returns an empty stream

            # With raise_on_failure=True
            fetcher = LinkContentFetcher(raise_on_failure=True)
            with pytest.raises(httpx.HTTPStatusError):
                await fetcher.run_async(urls=["https://www.example.com"])

    @pytest.mark.asyncio
    async def test_run_async_user_agent_rotation(self):
        """Test user agent rotation in async fetching"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.get") as mock_get:
            # First call raises an error to trigger user agent rotation
            first_response = Mock(status_code=403)
            first_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "403 Forbidden", request=Mock(), response=first_response
            )

            # Second call succeeds
            second_response = Mock(status_code=200, text="Success", headers={"Content-Type": "text/plain"})

            # Use side_effect to return different responses on consecutive calls
            mock_get.side_effect = [first_response, second_response]

            # Create fetcher with custom user agents
            fetcher = LinkContentFetcher(user_agents=["agent1", "agent2"], retry_attempts=1)

            # Should succeed on the second attempt with the second user agent
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]
            assert len(streams) == 1
            assert streams[0].data == b"Success"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_integration(self):
        """Test async fetching with real HTTP requests"""
        fetcher = LinkContentFetcher()
        streams = (await fetcher.run_async([HTML_URL]))["streams"]
        first_stream = streams[0]
        assert "Haystack" in first_stream.data.decode("utf-8")
        assert first_stream.meta["content_type"] == "text/html"
        assert first_stream.mime_type == "text/html"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_multiple_integration(self):
        """Test async fetching of multiple URLs with real HTTP requests"""
        fetcher = LinkContentFetcher()
        streams = (await fetcher.run_async([HTML_URL, TEXT_URL]))["streams"]
        assert len(streams) == 2

        for stream in streams:
            assert "Haystack" in stream.data.decode("utf-8")

            if stream.meta["url"] == HTML_URL:
                assert stream.meta["content_type"] == "text/html"
                assert stream.mime_type == "text/html"
            elif stream.meta["url"] == TEXT_URL:
                assert stream.meta["content_type"] == "text/plain"
                assert stream.mime_type == "text/plain"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_http2(self):
        """Test async fetching with HTTP/2 enabled"""
        # Mock the h2 import check so test works without h2 installed
        with patch("haystack.lazy_imports.LazyImport.check"):
            fetcher = LinkContentFetcher(http2=True)
            streams = (await fetcher.run_async([HTML_URL]))["streams"]
            assert len(streams) == 1
            assert "Haystack" in streams[0].data.decode("utf-8")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_http2_import_error(self):
        """Test async fetching with HTTP/2 enabled but h2 not installed"""
        # Mock the h2 import check to fail
        with patch("haystack.lazy_imports.LazyImport.check", side_effect=ImportError("No module named 'h2'")):
            fetcher = LinkContentFetcher(http2=True)
            # Verify http2 was disabled after import error
            assert fetcher.http2 is False
            # Verify we can still make requests
            streams = (await fetcher.run_async([HTML_URL]))["streams"]
            assert len(streams) == 1
            assert "Haystack" in streams[0].data.decode("utf-8")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_with_client_kwargs(self):
        """Test async fetching with custom client kwargs"""
        fetcher = LinkContentFetcher(client_kwargs={"follow_redirects": True, "timeout": 10.0})
        streams = (await fetcher.run_async([HTML_URL]))["streams"]
        assert len(streams) == 1
        assert "Haystack" in streams[0].data.decode("utf-8")
