# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tenacity import wait_none

from haystack.components.fetchers.link_content import (
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_USER_AGENT,
    LinkContentFetcher,
    ResponseTooLargeError,
    UnsafeTargetError,
    _binary_content_handler,
    _text_content_handler,
)
from haystack.core.serialization import component_from_dict, component_to_dict

HTML_URL = "https://docs.haystack.deepset.ai/docs/intro"
TEXT_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
PDF_URL = "https://raw.githubusercontent.com/deepset-ai/haystack/b5987a6d8d0714eb2f3011183ab40093d2e4a41a/e2e/samples/pipelines/sample_pdf_1.pdf"

PUBLIC_IP = "93.184.216.34"


def make_response(
    url: str,
    status_code: int = 200,
    text: str | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """
    Builds a real `httpx.Response` suitable for replaying from a mocked `Client.stream` call.
    """
    return httpx.Response(status_code, text=text, content=content, headers=headers, request=httpx.Request("GET", url))


def stream_side_effect(responses: list) -> "MagicMock":
    """
    Creates a `side_effect` for a mocked `httpx.Client.stream` that replays the given responses in order.

    Entries may be `httpx.Response` objects (replayed through a context manager) or exceptions (raised).
    The last entry is repeated once the list is exhausted.
    """
    state = {"idx": 0}

    def fake_stream(method: str, url: str, headers: dict | None = None, **kwargs):  # noqa: ARG001
        idx = state["idx"]
        state["idx"] = min(idx + 1, len(responses) - 1)
        item = responses[idx]
        if isinstance(item, Exception):
            raise item
        return nullcontext(item)

    return fake_stream


@pytest.fixture(autouse=True)
def mock_dns_resolution(request):
    """
    Points all name resolution performed by the fetcher at a public IP so unit tests never touch the network.

    Tests that exercise the IP validation install their own patch on top of this one.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    with patch("haystack.components.fetchers.link_content._resolve_host", return_value=[PUBLIC_IP]) as mocked:
        yield mocked


@pytest.fixture
def mock_get_link_text_content():
    with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
        mock_stream.side_effect = stream_side_effect(
            [make_response("https://www.example.com", text="Example test response")]
        )
        yield mock_stream


@pytest.fixture
def mock_get_link_content(test_files_path):
    with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
        with open(test_files_path / "pdf" / "sample_pdf_1.pdf", "rb") as f1:
            file_bytes = f1.read()
        mock_stream.side_effect = stream_side_effect(
            [make_response("https://www.example.com", content=file_bytes, headers={"Content-Type": "application/pdf"})]
        )
        yield mock_stream


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
        assert fetcher.allowed_hosts is None
        assert fetcher.max_response_bytes == DEFAULT_MAX_RESPONSE_BYTES
        assert fetcher.max_redirects == DEFAULT_MAX_REDIRECTS
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
        assert fetcher._client is None
        assert fetcher._async_client is None

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
        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [make_response("https://www.example.com", text="Example test response")]
            )
            fetcher = LinkContentFetcher()
            streams = fetcher.run(urls=["https://www.example.com"])["streams"]
            first_stream = streams[0]
            assert first_stream.data == correct_response
            assert first_stream.meta["content_type"] == "text/plain"
            assert first_stream.mime_type == "text/plain"

    def test_run_html(self):
        """Test fetching HTML content"""
        correct_response = b"<h1>Example test response</h1>"
        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response(
                        "https://www.example.com",
                        content=b"<h1>Example test response</h1>",
                        headers={"Content-Type": "text/html"},
                    )
                ]
            )
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
        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            response = make_response(
                "https://www.example.com", content=file_bytes, headers={"Content-Type": "application/pdf"}
            )
            mock_stream.side_effect = stream_side_effect([response])
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
        mock_response = make_response("https://www.example.com", status_code=403)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([mock_response])
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

        mock_response = make_response("https://non_existent_website_dot.com/", status_code=403)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([mock_response])
            with pytest.raises(httpx.HTTPStatusError):
                fetcher.run(["https://non_existent_website_dot.com/"])

    def test_run_retries_once_when_retry_attempts_is_one(self):
        url = "https://www.example.com"
        successful_response = make_response(url, text="Success")

        with patch("haystack.components.fetchers.link_content.httpx.Client") as client_mock:
            client = client_mock.return_value
            client.headers = {}
            client.stream.side_effect = stream_side_effect(
                [httpx.RequestError("transient failure", request=httpx.Request("GET", url)), successful_response]
            )

            fetcher = LinkContentFetcher(retry_attempts=1)
            with patch("haystack.components.fetchers.link_content.wait_exponential", return_value=wait_none()):
                streams = fetcher.run(urls=[url])["streams"]

        assert streams[0].data == successful_response.text.encode()
        assert client.stream.call_count == 2

    def test_request_headers_merging_and_ua_override(self):
        # Patch the Client class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock:
            client = ClientMock.return_value
            client.headers = {}  # base headers used in the merge
            mock_response = make_response("https://example.com", text="OK")
            client.stream.side_effect = stream_side_effect([mock_response])

            fetcher = LinkContentFetcher(
                user_agents=["ua-sync-1", "ua-sync-2"],
                request_headers={
                    "Accept-Language": "fr-FR",
                    "X-Test": "1",
                    "User-Agent": "will-be-overridden",  # rotating UA must override this
                },
            )

            _ = fetcher.run(urls=["https://example.com"])["streams"]

            client.stream.assert_called_once()
            sent_headers = client.stream.call_args.kwargs["headers"]
            assert sent_headers["X-Test"] == "1"
            assert sent_headers["Accept-Language"] == "fr-FR"
            assert sent_headers["User-Agent"] == "ua-sync-1"  # rotating UA wins

    def test_user_agent_rotation_is_independent_per_url(self):
        """
        Every URL in a `run` call retries on its own, so every URL must walk its own user agent list.

        The rotation cursor used to live on the component, and `run` fetches the URLs concurrently, so the
        cursor was advanced and reset by whichever fetches happened to be in flight at the same time.
        """
        urls = [f"https://example.com/{i}" for i in range(8)]
        user_agents = [f"ua-{i}" for i in range(4)]

        attempts: dict[str, int] = {}
        user_agent_on_success: dict[str, str] = {}
        lock = threading.Lock()

        def fake_stream(method, url, headers=None, **kwargs):  # noqa: ARG001
            with lock:
                attempt = attempts.get(url, 0)
                attempts[url] = attempt + 1
            if attempt == 0:
                # Every URL fails once, so every URL rotates once.
                raise httpx.RequestError("simulated transient failure", request=httpx.Request("GET", url))
            with lock:
                user_agent_on_success[url] = headers["User-Agent"]
            return nullcontext(make_response(url, text="OK"))

        with patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock:
            client = ClientMock.return_value
            client.headers = {}
            client.stream.side_effect = fake_stream

            fetcher = LinkContentFetcher(user_agents=user_agents, retry_attempts=3, raise_on_failure=False)
            with patch("haystack.components.fetchers.link_content.wait_exponential", return_value=wait_none()):
                fetcher.run(urls=urls)

        # Each URL failed once and succeeded on its first retry, so each one sends the second user agent.
        assert user_agent_on_success == dict.fromkeys(urls, user_agents[1])


class TestComponentLifecycle:
    def test_clients_are_none_after_init(self):
        fetcher = LinkContentFetcher()
        assert fetcher._client is None
        assert fetcher._async_client is None

    def test_sync_lifecycle(self):
        with patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock:
            client_instance = ClientMock.return_value
            fetcher = LinkContentFetcher()

            fetcher.warm_up()
            assert fetcher._client is client_instance
            assert fetcher._async_client is None
            ClientMock.assert_called_once()

            fetcher.close()
            client_instance.close.assert_called_once()
            assert fetcher._client is None

    def test_warm_up_is_idempotent(self):
        with patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock:
            fetcher = LinkContentFetcher()
            fetcher.warm_up()
            fetcher.warm_up()
            ClientMock.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_lifecycle(self):
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            async_client_instance = AsyncClientMock.return_value
            async_client_instance.aclose = AsyncMock()
            fetcher = LinkContentFetcher()

            await fetcher.warm_up_async()
            assert fetcher._async_client is async_client_instance
            assert fetcher._client is None
            AsyncClientMock.assert_called_once()

            await fetcher.close_async()
            async_client_instance.aclose.assert_awaited_once()
            assert fetcher._async_client is None

    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self):
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            fetcher = LinkContentFetcher()
            await fetcher.warm_up_async()
            await fetcher.warm_up_async()
            AsyncClientMock.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        fetcher = LinkContentFetcher()
        fetcher.close()
        await fetcher.close_async()
        assert fetcher._client is None
        assert fetcher._async_client is None

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        with (
            patch("haystack.components.fetchers.link_content.httpx.Client") as ClientMock,
            patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock,
        ):
            client_instance = ClientMock.return_value
            async_client_instance = AsyncClientMock.return_value
            async_client_instance.aclose = AsyncMock()

            fetcher = LinkContentFetcher()
            fetcher.warm_up()
            await fetcher.warm_up_async()

            fetcher.close()
            assert fetcher._client is None
            assert fetcher._async_client is async_client_instance
            async_client_instance.aclose.assert_not_awaited()

            await fetcher.close_async()
            assert fetcher._async_client is None
            client_instance.close.assert_called_once()

    def test_run_self_heals(self):
        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", text="ok")])
            fetcher = LinkContentFetcher()
            fetcher.run(urls=["https://www.example.com"])
            assert fetcher._client is not None

    @pytest.mark.asyncio
    async def test_run_async_self_heals(self):
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", text="ok")])
            fetcher = LinkContentFetcher()
            await fetcher.run_async(urls=["https://www.example.com"])
            assert fetcher._async_client is not None


@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.integration
class TestLinkContentFetcherIntegration:
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


@pytest.mark.asyncio
class TestLinkContentFetcherAsync:
    async def test_run_async(self):
        """Test basic async fetching with a mocked response"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [make_response("https://www.example.com", text="Example test response")]
            )

            fetcher = LinkContentFetcher()
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]

            first_stream = streams[0]
            expected_content = b"Example test response"
            assert first_stream.data == expected_content
            assert first_stream.meta["content_type"] == "text/plain"
            assert first_stream.mime_type == "text/plain"

    async def test_run_async_multiple(self):
        """Test async fetching of multiple URLs with mocked responses"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [make_response("https://www.example.com", text="Example test response")]
            )

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

    async def test_run_async_empty_urls(self):
        """Test async fetching with empty URL list"""
        fetcher = LinkContentFetcher()
        streams = (await fetcher.run_async(urls=[]))["streams"]
        assert len(streams) == 0

    async def test_run_async_error_handling(self):
        """Test error handling for async fetching"""
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", status_code=404)])

            # With raise_on_failure=False
            fetcher = LinkContentFetcher(raise_on_failure=False, retry_attempts=0)
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]
            assert len(streams) == 1  # Returns an empty stream

            # With raise_on_failure=True
            fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=0)
            with pytest.raises(httpx.HTTPStatusError):
                await fetcher.run_async(urls=["https://www.example.com"])

    async def test_run_async_user_agent_rotation(self):
        """Test user agent rotation in async fetching"""
        with (
            patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream,
            patch("asyncio.sleep") as mock_sleep,
        ):
            # Mock asyncio.sleep used by tenacity to keep this test fast
            mock_sleep.return_value = None

            # First call raises an error to trigger user agent rotation
            first_response = make_response("https://www.example.com", status_code=403)

            # Second call succeeds
            second_response = make_response(
                "https://www.example.com", text="Success", headers={"Content-Type": "text/plain"}
            )

            # Use side_effect to return different responses on consecutive calls
            mock_stream.side_effect = stream_side_effect([first_response, second_response])

            # Create fetcher with custom user agents
            fetcher = LinkContentFetcher(user_agents=["agent1", "agent2"], retry_attempts=1)

            # Should succeed on the second attempt with the second user agent
            streams = (await fetcher.run_async(urls=["https://www.example.com"]))["streams"]
            assert len(streams) == 1
            expected_result = b"Success"
            assert streams[0].data == expected_result

            mock_sleep.assert_called_once()

    async def test_request_headers_merging_and_ua_override(self):
        # Patch the AsyncClient class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            aclient = AsyncClientMock.return_value
            aclient.headers = {}  # base headers used in the merge

            mock_response = make_response("https://example.com", text="OK")
            aclient.stream = MagicMock()
            aclient.stream.side_effect = stream_side_effect([mock_response])

            fetcher = LinkContentFetcher(
                user_agents=["ua-async-1", "ua-async-2"],
                request_headers={"Accept-Language": "de-DE", "X-Async": "true", "User-Agent": "ignored-here-too"},
            )

            _ = (await fetcher.run_async(urls=["https://example.com"]))["streams"]

            assert aclient.stream.call_count == 1
            sent_headers = aclient.stream.call_args.kwargs["headers"]
            assert sent_headers["X-Async"] == "true"
            assert sent_headers["Accept-Language"] == "de-DE"
            assert sent_headers["User-Agent"] == "ua-async-1"  # rotating UA wins

    async def test_duplicated_request_headers_merging(self):
        # Patch the AsyncClient class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            aclient = AsyncClientMock.return_value
            aclient.headers = {}  # base headers used in the merge

            mock_response = make_response("https://example.com", text="OK")
            aclient.stream = MagicMock()
            aclient.stream.side_effect = stream_side_effect([mock_response])

            fetcher = LinkContentFetcher(
                request_headers={
                    "x-test-header": "header-1",
                    "X-Test-Header": "agent-2",
                    "X-TEST-HEADER": "agent-3",
                    "X-TeSt-HeAdEr": "good-one",
                }
            )

            _ = (await fetcher.run_async(urls=["https://example.com"]))["streams"]

            assert aclient.stream.call_count == 1
            sent_headers = aclient.stream.call_args.kwargs["headers"]
            existing_keys = {}
            for key, value in sent_headers.items():
                lower_key = key.lower()
                if lower_key in existing_keys:
                    raise AssertionError()
                if lower_key == "x-test-header":
                    assert value == "good-one"
                existing_keys[lower_key] = key

            assert "x-test-header" in existing_keys
            assert existing_keys["x-test-header"] == "X-TeSt-HeAdEr"


class TestLinkContentFetcherSecurity:
    """
    Tests for the SSRF protections: the `allowed_hosts` whitelist, rejection of forbidden IP ranges,
    per-hop redirect validation, and the response size cap.
    """

    @pytest.mark.parametrize(
        "forbidden_ip",
        [
            "10.0.0.5",
            "192.168.1.10",
            "172.16.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "224.0.0.1",
            "::1",
            "fd00::1",
            "fe80::1",
            "100.64.0.1",
        ],
    )
    def test_forbidden_ip_targets_are_rejected(self, mock_dns_resolution, forbidden_ip):
        """Hosts resolving to private/internal ranges must be rejected before any request is made."""
        mock_dns_resolution.return_value = [forbidden_ip]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                fetcher.run(["https://internal.example.com/data"])

        mock_stream.assert_not_called()

    @pytest.mark.parametrize("url", ["http://127.0.0.1:8080/admin", "http://10.0.0.5/", "http://[::1]/"])
    def test_ip_literal_urls_are_rejected(self, mock_dns_resolution, url):
        """URLs pointing directly at an internal IP literal must be rejected.

        Name resolution echoes IP literals back unchanged, like a real resolver does.
        """
        mock_dns_resolution.side_effect = lambda host, port: [host]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                fetcher.run([url])

        mock_stream.assert_not_called()

    def test_host_resolving_to_mixed_addresses_is_rejected(self, mock_dns_resolution):
        """If any of the resolved addresses is forbidden, the whole host is rejected."""
        mock_dns_resolution.return_value = ["93.184.216.34", "192.168.0.1"]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with pytest.raises(UnsafeTargetError, match="192.168.0.1"):
            fetcher.run(["https://example.com"])

    def test_allowed_hosts_whitelist(self, mock_dns_resolution):
        """Hosts outside the whitelist are rejected; subdomains of allowed entries are accepted."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(allowed_hosts=["example.com"], retry_attempts=0)
        expected_data = b"ok"

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://api.example.com", text="ok")])
            streams = fetcher.run(["https://api.example.com"])["streams"]
            assert streams[0].data == expected_data

            with pytest.raises(UnsafeTargetError, match="whitelist"):
                fetcher.run(["https://api.other.com"])

            # the suffix match must respect domain boundaries
            with pytest.raises(UnsafeTargetError, match="whitelist"):
                fetcher.run(["https://notexample.com"])

    def test_no_allowlist_keeps_previous_behavior(self, mock_dns_resolution):
        """With the default configuration (no allowlist), any host resolving to a public IP can be fetched."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher()
        expected_data = b"ok"

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://anything.example.org", text="ok")])
            streams = fetcher.run(["https://anything.example.org"])["streams"]
            assert streams[0].data == expected_data

    def test_redirect_to_forbidden_target_is_rejected(self, mock_dns_resolution):
        """A redirect from a trusted host to an internal address must never be fetched."""
        mock_dns_resolution.side_effect = lambda host, port: (
            ["10.0.0.5"] if host == "internal.example.net" else [PUBLIC_IP]
        )
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response(
                        "https://www.example.com",
                        status_code=302,
                        headers={"location": "http://internal.example.net/secret"},
                    )
                ]
            )
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                fetcher.run(["https://www.example.com"])

            # only the first, legitimate hop was requested
            assert mock_stream.call_count == 1

    def test_redirect_to_ip_literal_is_rejected(self, mock_dns_resolution):
        """A redirect pointing directly at an internal IP literal must never be fetched."""
        mock_dns_resolution.side_effect = lambda host, port: [host] if host == "127.0.0.1" else [PUBLIC_IP]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response(
                        "https://www.example.com", status_code=301, headers={"location": "http://127.0.0.1:8080/"}
                    )
                ]
            )
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                fetcher.run(["https://www.example.com"])

            assert mock_stream.call_count == 1

    def test_redirects_are_followed_and_validated(self, mock_dns_resolution):
        """Legitimate redirects, including relative ones, are followed hop by hop."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(retry_attempts=0)
        expected_data = b"final content"

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response("https://www.example.com/a", status_code=302, headers={"location": "/b"}),
                    make_response(
                        "https://www.example.com/b", status_code=301, headers={"location": "https://cdn.example.org/c"}
                    ),
                    make_response("https://cdn.example.org/c", text="final content"),
                ]
            )
            streams = fetcher.run(["https://www.example.com/a"])["streams"]

            assert streams[0].data == expected_data
            assert streams[0].meta["url"] == "https://www.example.com/a"
            assert mock_stream.call_count == 3
            # every hop went through the same validated request path
            requested_urls = [call.args[1] for call in mock_stream.call_args_list]
            assert requested_urls == [
                "https://www.example.com/a",
                "https://www.example.com/b",
                "https://cdn.example.org/c",
            ]

    def test_too_many_redirects(self, mock_dns_resolution):
        """Redirect chains longer than `max_redirects` raise `httpx.TooManyRedirects`."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(max_redirects=2, retry_attempts=0)

        redirects = [
            make_response(
                f"https://www.example.com/{i}",
                status_code=302,
                headers={"location": f"https://www.example.com/{i + 1}"},
            )
            for i in range(10)
        ]
        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(redirects)
            with pytest.raises(httpx.TooManyRedirects, match="Exceeded 2 redirects"):
                fetcher.run(["https://www.example.com/0"])

            # the original request plus `max_redirects` followed hops
            assert mock_stream.call_count == 3

    def test_redirects_not_followed_when_disabled_in_client_kwargs(self, mock_dns_resolution):
        """`follow_redirects=False` in `client_kwargs` keeps the pre-existing behavior: the redirect is not
        followed and its response surfaces like an error, exactly as httpx did before."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(client_kwargs={"follow_redirects": False}, retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response(
                        "https://www.example.com", status_code=302, content=b"moved", headers={"location": "/b"}
                    )
                ]
            )
            with pytest.raises(httpx.HTTPStatusError, match="302"):
                fetcher.run(["https://www.example.com"])

            assert mock_stream.call_count == 1

    @pytest.mark.parametrize("raise_on_failure", [True, False])
    def test_response_size_limit(self, mock_dns_resolution, raise_on_failure):
        """Response bodies larger than `max_response_bytes` raise `ResponseTooLargeError`."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(max_response_bytes=10, retry_attempts=0, raise_on_failure=raise_on_failure)
        expected_empty = b""

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", content=b"a" * 11)])
            if raise_on_failure:
                with pytest.raises(ResponseTooLargeError, match="exceeds max_response_bytes=10"):
                    fetcher.run(["https://www.example.com"])
            else:
                # failures are swallowed and an empty stream is returned, like any other fetch error
                streams = fetcher.run(["https://www.example.com"])["streams"]
                assert len(streams) == 1
                assert streams[0].data == expected_empty

    def test_response_at_size_limit_is_fetched(self, mock_dns_resolution):
        """A body exactly of `max_response_bytes` is fetched successfully."""
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(max_response_bytes=10, retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.Client.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", content=b"a" * 10)])
            streams = fetcher.run(["https://www.example.com"])["streams"]
            assert streams[0].data == b"a" * 10

    def test_serialization_includes_new_init_parameters(self):
        fetcher = LinkContentFetcher()
        data = component_to_dict(fetcher, "fetcher")
        assert data["init_parameters"]["allowed_hosts"] is None
        assert data["init_parameters"]["max_response_bytes"] == DEFAULT_MAX_RESPONSE_BYTES
        assert data["init_parameters"]["max_redirects"] == DEFAULT_MAX_REDIRECTS
        # the manual redirect following is an internal detail: it must not leak into the serialized state
        assert data["init_parameters"]["client_kwargs"] == {"timeout": 3}

        fetcher = LinkContentFetcher(allowed_hosts=["example.com"], max_response_bytes=1024, max_redirects=2)
        data = component_to_dict(fetcher, "fetcher")
        assert data["init_parameters"]["allowed_hosts"] == ["example.com"]
        assert data["init_parameters"]["max_response_bytes"] == 1024
        assert data["init_parameters"]["max_redirects"] == 2

    def test_client_kwargs_serialization_round_trip(self, mock_dns_resolution):
        """A fetcher serialized and loaded back keeps following redirects."""
        fetcher = LinkContentFetcher()
        data = component_to_dict(fetcher, "fetcher")
        restored = component_from_dict(LinkContentFetcher, data, "fetcher")

        assert restored._follow_redirects is True
        assert restored.client_kwargs.get("timeout") == 3


@pytest.mark.asyncio
class TestLinkContentFetcherAsyncSecurity:
    async def test_run_async_rejects_forbidden_ip(self, mock_dns_resolution):
        mock_dns_resolution.return_value = ["10.0.0.5"]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                await fetcher.run_async(["https://internal.example.com/data"])

        mock_stream.assert_not_called()

    async def test_run_async_redirect_to_forbidden_target_is_rejected(self, mock_dns_resolution):
        mock_dns_resolution.side_effect = lambda host, port: ["192.168.0.1"] if host == "192.168.0.1" else [PUBLIC_IP]
        fetcher = LinkContentFetcher(retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response(
                        "https://www.example.com", status_code=302, headers={"location": "http://192.168.0.1/admin"}
                    )
                ]
            )
            with pytest.raises(UnsafeTargetError, match="forbidden IP address"):
                await fetcher.run_async(["https://www.example.com"])

            assert mock_stream.call_count == 1

    async def test_run_async_response_size_limit(self, mock_dns_resolution):
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(max_response_bytes=10, retry_attempts=0)

        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect([make_response("https://www.example.com", content=b"a" * 20)])
            with pytest.raises(ResponseTooLargeError, match="exceeds max_response_bytes=10"):
                await fetcher.run_async(["https://www.example.com"])

    async def test_run_async_follows_validated_redirects(self, mock_dns_resolution):
        mock_dns_resolution.return_value = [PUBLIC_IP]
        fetcher = LinkContentFetcher(retry_attempts=0)
        expected_data = b"final content"

        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient.stream") as mock_stream:
            mock_stream.side_effect = stream_side_effect(
                [
                    make_response("https://www.example.com/a", status_code=302, headers={"location": "/b"}),
                    make_response("https://www.example.com/b", text="final content"),
                ]
            )
            streams = (await fetcher.run_async(["https://www.example.com/a"]))["streams"]

            assert streams[0].data == expected_data
            assert mock_stream.call_count == 2


@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.integration
@pytest.mark.asyncio
class TestLinkContentFetcherAsyncIntegration:
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

    async def test_run_async_with_client_kwargs(self):
        """Test async fetching with custom client kwargs"""
        fetcher = LinkContentFetcher(client_kwargs={"follow_redirects": True, "timeout": 10.0})
        streams = (await fetcher.run_async([HTML_URL]))["streams"]
        assert len(streams) == 1
        assert "Haystack" in streams[0].data.decode("utf-8")
