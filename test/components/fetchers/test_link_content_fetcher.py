# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from haystack.components.fetchers.link_content import (
    DEFAULT_USER_AGENT,
    LinkContentFetcher,
    _binary_content_handler,
    _text_content_handler,
)

HTML_URL = "https://docs.haystack.deepset.ai/docs"
TEXT_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
PDF_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/b5987a6d8d0714eb2f3011183ab40093d2e4a41a/e2e/samples/pipelines/sample_pdf_1.pdf"


@pytest.fixture
def mock_get_link_text_content():
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        mock_response = Mock(status_code=200, text="Example test response", headers={"Content-Type": "text/plain"})
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_get_link_content(test_files_path):
    with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
        with open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb") as f1:
            file_bytes = f1.read()
        mock_response = Mock(status_code=200, content=file_bytes, headers={"Content-Type": "application/pdf"})
        mock_get.return_value = mock_response
        yield mock_get


class TestLinkContentFetcher:
    def test_init(self):
        """Test initialization with default parameters"""
        fetcher = LinkContentFetcher()
        assert fetcher.raise_on_failure is True
        assert fetcher.user_agents == [DEFAULT_USER_AGENT]
        assert fetcher.retry_attempts == 2
        assert fetcher.timeout == 3
        assert fetcher.http2 is False
        assert isinstance(fetcher.client_kwargs, dict)
        assert fetcher.handlers == {
            "text/*": _text_content_handler,
            "text/html": _binary_content_handler,
            "application/json": _text_content_handler,
            "application/*": _binary_content_handler,
            "image/*": _binary_content_handler,
            "audio/*": _binary_content_handler,
            "video/*": _binary_content_handler,
        }
        assert hasattr(fetcher, "_get_response")
        assert hasattr(fetcher, "_client")
        assert isinstance(fetcher._client, httpx.Client)

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        fetcher = LinkContentFetcher(
            raise_on_failure=False,
            user_agents=["test"],
            retry_attempts=1,
            timeout=2,
            http2=True,
            client_kwargs={"verify": False},
        )
        assert fetcher.raise_on_failure is False
        assert fetcher.user_agents == ["test"]
        assert fetcher.retry_attempts == 1
        assert fetcher.timeout == 2
        assert fetcher.http2 is True
        assert "verify" in fetcher.client_kwargs
        assert fetcher.client_kwargs["verify"] is False

    def test_run_text(self):
        """Test fetching text content"""
        correct_response = b"Example test response"
        with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200, text="Example test response", headers={"Content-Type": "text/plain"})
            mock_get.return_value = mock_response
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == correct_response
            assert first_stream.meta["content_type"] == "text/plain"
            assert first_stream.mime_type == "text/plain"

    def test_run_html(self):
        """Test fetching HTML content"""
        correct_response = b"<h1>Example test response</h1>"
        with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
            mock_response = Mock(
                status_code=200, content=b"<h1>Example test response</h1>", headers={"Content-Type": "text/html"}
            )
            mock_get.return_value = mock_response
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == correct_response
            assert first_stream.meta["content_type"] == "text/html"
            assert first_stream.mime_type == "text/html"

    def test_run_binary(self, test_files_path):
        """Test fetching binary content"""
        with open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb") as f1:
            file_bytes = f1.read()
        with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
            mock_response = Mock(status_code=200, content=file_bytes, headers={"Content-Type": "application/pdf"})
            mock_get.return_value = mock_response
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == file_bytes
            assert first_stream.meta["content_type"] == "application/pdf"
            assert first_stream.mime_type == "application/pdf"

    def test_run_bad_request_no_exception(self):
        """Test behavior when a request results in an error status code"""
        empty_byte_stream = b""
        fetcher = LinkContentFetcher(raise_on_failure=False, retry_attempts=0)
        mock_response = Mock(status_code=403)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Client Error", request=Mock(), response=mock_response
        )

        with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
            mock_get.return_value = mock_response
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]

        # empty byte stream is returned because raise_on_failure is False
        assert len(streams) == 1
        first_stream = streams[0]
        assert first_stream.data == empty_byte_stream
        assert first_stream.meta["content_type"] == "text/html"
        assert first_stream.mime_type == "text/html"

    def test_bad_request_exception_raised(self):
        """
        This test is to ensure that the fetcher raises an exception when a single bad request is made and it is
        configured to do so.
        """
        fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=0)

        mock_response = Mock(status_code=403)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Client Error", request=Mock(), response=mock_response
        )

        with patch("haystack.components.fetchers.link_content.httpx.Client.get") as mock_get:
            mock_get.return_value = mock_response
            with pytest.raises(httpx.HTTPStatusError):
                fetcher.run(["https://non_existent_website_dot.com/"])

    def test_request_headers_merging_and_ua_override(self):
        # Patch the Client class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock:
            client = ClientMock.return_value
            client.headers = {}  # base headers used in the merge
            mock_response = Mock(status_code=200, text="OK", headers={"Content-Type": "text/plain"})
            client.get.return_value = mock_response

            fetcher = LinkContentFetcher(
                user_agents=["ua-sync-1", "ua-sync-2"],
                request_headers={
                    "Accept-Language": "fr-FR",
                    "X-Test": "1",
                    "User-Agent": "will-be-overridden",  # rotating UA must override this
                },
            )

            _ = fetcher.run(urls=["https://example.com"])["streams"]

            client.get.assert_called_once()
            sent_headers = client.get.call_args.kwargs["headers"]
            assert sent_headers["X-Test"] == "1"
            assert sent_headers["Accept-Language"] == "fr-FR"
            assert sent_headers["User-Agent"] == "ua-sync-1"  # rotating UA wins

    @pytest.mark.integration
    def test_link_content_fetcher_html(self):
        """
        Test fetching HTML content from a real URL.
        """
        fetcher = LinkContentFetcher()
        streams = fetcher.run([HTML_URL])["streams"]
        first_stream = streams[0]
        assert "Haystack" in first_stream.data.decode("utf-8")
        assert first_stream.meta["content_type"] == "text/html"
        assert "url" in first_stream.meta and first_stream.meta["url"] == HTML_URL
        assert first_stream.mime_type == "text/html"

    @pytest.mark.integration
    def test_link_content_fetcher_text(self):
        """
        Test fetching text content from a real URL.
        """
        fetcher = LinkContentFetcher()
        streams = fetcher.run([TEXT_URL])["streams"]
        first_stream = streams[0]
        assert "Haystack" in first_stream.data.decode("utf-8")
        assert first_stream.meta["content_type"] == "text/plain"
        assert "url" in first_stream.meta and first_stream.meta["url"] == TEXT_URL
        assert first_stream.mime_type == "text/plain"

    @pytest.mark.integration
    def test_link_content_fetcher_multiple_different_content_types(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain different content types.
        """
        fetcher = LinkContentFetcher()
        streams = fetcher.run([PDF_URL, HTML_URL])["streams"]
        assert len(streams) == 2
        for stream in streams:
            assert stream.meta["content_type"] in ("text/html", "application/pdf", "application/octet-stream")
            if stream.meta["content_type"] == "text/html":
                assert "Haystack" in stream.data.decode("utf-8")
                assert stream.mime_type == "text/html"
            elif stream.meta["content_type"] == "application/pdf":
                assert len(stream.data) > 0
                assert stream.mime_type == "application/pdf"

    @pytest.mark.integration
    def test_link_content_fetcher_multiple_html_streams(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain different content types,
        and that we have two html streams.
        """

        fetcher = LinkContentFetcher()
        streams = fetcher.run([PDF_URL, HTML_URL, "https://google.com"])["streams"]
        assert len(streams) == 3
        for stream in streams:
            assert stream.meta["content_type"] in ("text/html", "application/pdf", "application/octet-stream")
            if stream.meta["content_type"] == "text/html":
                assert "Haystack" in stream.data.decode("utf-8") or "Google" in stream.data.decode("utf-8")
                assert stream.mime_type == "text/html"
            elif stream.meta["content_type"] == "application/pdf":
                assert len(stream.data) > 0
                assert stream.mime_type == "application/pdf"

    @pytest.mark.integration
    def test_mix_of_good_and_failed_requests(self):
        """
        This test is to ensure that the fetcher can handle a list of URLs that contain URLs that fail to be fetched.
        In such a case, the fetcher should return the content of the URLs that were successfully fetched and not raise
        an exception.
        """
        fetcher = LinkContentFetcher(retry_attempts=0)
        result = fetcher.run(["https://non_existent_website_dot.com/", "https://www.google.com/"])
        assert len(result["streams"]) == 1
        first_stream = result["streams"][0]
        assert first_stream.meta["content_type"] == "text/html"
        assert first_stream.mime_type == "text/html"


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
            expected_content = b"Example test response"
            assert first_stream.data == expected_content
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
                expected_data = b"Example test response"
                assert stream.data == expected_data
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
            fetcher = LinkContentFetcher(raise_on_failure=False, retry_attempts=0)
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]
            assert len(streams) == 1  # Returns an empty stream

            # With raise_on_failure=True
            fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=0)
            with pytest.raises(httpx.HTTPStatusError):
                await fetcher.run_async(urls=["https://www.example.com"])

    @pytest.mark.asyncio
    async def test_run_async_user_agent_rotation(self):
        """Test user agent rotation in async fetching"""
        with (
            patch("haystack.components.fetchers.link_content.httpx.AsyncClient.get") as mock_get,
            patch("asyncio.sleep") as mock_sleep,
        ):
            # Mock asyncio.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

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
            expected_result = b"Success"
            assert streams[0].data == expected_result

            mock_sleep.assert_called_once()

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
    async def test_run_async_with_client_kwargs(self):
        """Test async fetching with custom client kwargs"""
        fetcher = LinkContentFetcher(client_kwargs={"follow_redirects": True, "timeout": 10.0})
        streams = (await fetcher.run_async([HTML_URL]))["streams"]
        assert len(streams) == 1
        assert "Haystack" in streams[0].data.decode("utf-8")

    @pytest.mark.asyncio
    async def test_request_headers_merging_and_ua_override(self):
        # Patch the AsyncClient class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            aclient = AsyncClientMock.return_value
            aclient.headers = {}  # base headers used in the merge

            mock_response = Mock(status_code=200, text="OK", headers={"Content-Type": "text/plain"})
            aclient.get = AsyncMock(return_value=mock_response)

            fetcher = LinkContentFetcher(
                user_agents=["ua-async-1", "ua-async-2"],
                request_headers={"Accept-Language": "de-DE", "X-Async": "true", "User-Agent": "ignored-here-too"},
            )

            _ = (await fetcher.run_async(urls=["https://example.com"]))["streams"]

            assert aclient.get.await_count == 1
            sent_headers = aclient.get.call_args.kwargs["headers"]
            assert sent_headers["X-Async"] == "true"
            assert sent_headers["Accept-Language"] == "de-DE"
            assert sent_headers["User-Agent"] == "ua-async-1"  # rotating UA wins
