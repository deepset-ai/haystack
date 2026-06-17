# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from haystack.components.fetchers.link_content import (
    DEFAULT_USER_AGENT,
    LinkContentFetcher,
    UnsafeFetchURLError,
    _assert_safe_url,
    _binary_content_handler,
    _is_blocked_ip,
    _ssrf_guard_request_hook,
    _ssrf_guard_request_hook_async,
    _text_content_handler,
)

HTML_URL = "https://docs.haystack.deepset.ai/docs/intro"
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

    async def test_run_async_empty_urls(self):
        """Test async fetching with empty URL list"""
        fetcher = LinkContentFetcher()
        streams = (await fetcher.run_async(urls=[]))["streams"]
        assert len(streams) == 0

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

    async def test_duplicated_request_headers_merging(self):
        # Patch the AsyncClient class to control the instance created by LinkContentFetcher
        with patch("haystack.components.fetchers.link_content.httpx.AsyncClient") as AsyncClientMock:
            aclient = AsyncClientMock.return_value
            aclient.headers = {}  # base headers used in the merge

            mock_response = Mock(status_code=200, text="OK", headers={"Content-Type": "text/plain"})
            aclient.get = AsyncMock(return_value=mock_response)

            fetcher = LinkContentFetcher(
                request_headers={
                    "x-test-header": "header-1",
                    "X-Test-Header": "agent-2",
                    "X-TEST-HEADER": "agent-3",
                    "X-TeSt-HeAdEr": "good-one",
                }
            )

            _ = (await fetcher.run_async(urls=["https://example.com"]))["streams"]

            assert aclient.get.await_count == 1
            sent_headers = aclient.get.call_args.kwargs["headers"]
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


class TestLinkContentFetcherSSRFGuard:
    """
    Tests for the Server-Side Request Forgery (SSRF) guard.

    LinkContentFetcher can be exposed to an LLM as a tool (via ComponentTool), which lets model-controlled
    arguments choose the fetched URLs. Without a guard, those URLs could target loopback, private-network,
    link-local, or cloud-metadata addresses. These tests assert the request is blocked *before* any network
    egress and that the existing error-handling and opt-out behavior is preserved.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/health",
            "http://127.0.0.1:8080/health",
            "http://localhost/admin",
            "http://0.0.0.0/",
            "http://10.0.0.5/internal",
            "http://192.168.1.10/router",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
            "ftp://example.com/secret",  # disallowed scheme
            "file:///etc/passwd",  # disallowed scheme
        ],
    )
    def test_assert_safe_url_blocks_non_routable_targets(self, url):
        """The guard rejects loopback / private / link-local / metadata / bad-scheme URLs."""
        with pytest.raises(UnsafeFetchURLError):
            _assert_safe_url(url)

    @pytest.mark.parametrize(
        "ip", ["127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.5.5", "169.254.169.254", "0.0.0.0", "::1", "fc00::1"]
    )
    def test_is_blocked_ip_true_for_non_routable(self, ip):
        import ipaddress

        assert _is_blocked_ip(ipaddress.ip_address(ip)) is True

    @pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2001:4860:4860::8888"])
    def test_is_blocked_ip_false_for_public(self, ip):
        import ipaddress

        assert _is_blocked_ip(ipaddress.ip_address(ip)) is False

    def test_run_blocks_localhost_and_raises(self):
        """A model-provided localhost URL must be blocked before any network egress."""
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must never be reached
            sent.append(str(request.url))
            return httpx.Response(200, text="SECRET")

        fetcher = LinkContentFetcher(retry_attempts=0)
        # Preserve the SSRF event hooks but route any *actual* request to a recording mock transport,
        # so a failure to block would be observable as a recorded request instead of real egress.
        fetcher._client = httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=True, event_hooks=fetcher._client.event_hooks
        )
        with pytest.raises(UnsafeFetchURLError):
            fetcher.run(urls=["http://127.0.0.1:80/health"])
        assert sent == []  # the guard fired before the transport was reached

    def test_run_blocks_host_resolving_to_metadata_ip(self):
        """A public-looking host that resolves to the cloud metadata IP (DNS rebinding) is blocked."""
        sent = []

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must never be reached
            sent.append(str(request.url))
            return httpx.Response(200, text="SECRET")

        with patch("haystack.components.fetchers.link_content.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [(2, 1, 6, "", ("169.254.169.254", 80))]
            fetcher = LinkContentFetcher(retry_attempts=0)
            fetcher._client = httpx.Client(
                transport=httpx.MockTransport(handler), follow_redirects=True, event_hooks=fetcher._client.event_hooks
            )
            with pytest.raises(UnsafeFetchURLError):
                fetcher.run(urls=["http://attacker-controlled.example/x"])
            assert sent == []  # blocked before any connection attempt

    def test_run_blocked_url_with_raise_on_failure_false_returns_empty(self):
        """With raise_on_failure=False a blocked URL is skipped gracefully (empty stream), not raised."""
        fetcher = LinkContentFetcher(raise_on_failure=False, retry_attempts=0)
        result = fetcher.run(urls=["http://10.0.0.1/internal"])
        assert len(result["streams"]) == 1
        empty = b""
        assert result["streams"][0].data == empty

    def test_run_blocks_redirect_to_private_address(self):
        """A public URL that redirects to a private/metadata address is blocked on the redirect hop."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "public.example":
                return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
            return httpx.Response(200, text="SECRET")  # pragma: no cover - must never be reached

        fetcher = LinkContentFetcher(retry_attempts=0)
        # Swap in a mock transport while preserving the SSRF event hooks installed on the client.
        fetcher._client = httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=True, event_hooks=fetcher._client.event_hooks
        )
        with pytest.raises(UnsafeFetchURLError):
            fetcher.run(urls=["http://public.example/start"])

    def test_allow_private_addresses_opts_out_of_guard(self):
        """allow_private_addresses=True disables the guard so internal hosts can be reached intentionally."""
        fetcher = LinkContentFetcher(allow_private_addresses=True, retry_attempts=0)
        assert fetcher.allow_private_addresses is True
        # No SSRF "request" event hook should be installed on either client.
        assert _ssrf_hook_count(fetcher._client) == 0
        assert _ssrf_hook_count(fetcher._async_client) == 0

        # With the guard disabled, a localhost request reaches httpx.Client.get (here mocked) instead of
        # being blocked by UnsafeFetchURLError.
        with patch.object(fetcher._client, "get") as mock_get:
            mock_get.return_value = Mock(
                status_code=200, text="ok", headers={"Content-Type": "text/plain"}, content=b"ok"
            )
            result = fetcher.run(urls=["http://127.0.0.1:8080/health"])
            mock_get.assert_called_once()
            assert len(result["streams"]) == 1

    def test_default_installs_guard_hook(self):
        """By default (allow_private_addresses=False) the SSRF guard is installed on both clients."""
        fetcher = LinkContentFetcher()
        assert fetcher.allow_private_addresses is False
        assert _ssrf_hook_count(fetcher._client) == 1
        assert _ssrf_hook_count(fetcher._async_client) == 1

    def test_user_event_hooks_are_preserved(self):
        """User-supplied request event hooks are kept and run after the SSRF guard."""
        user_hook = Mock()
        fetcher = LinkContentFetcher(client_kwargs={"event_hooks": {"request": [user_hook]}})
        request_hooks = fetcher._client.event_hooks.get("request", [])
        assert user_hook in request_hooks
        # Guard runs first, user hook second.
        assert len(request_hooks) == 2
        assert request_hooks[-1] is user_hook

    async def test_run_async_blocks_localhost(self):
        """The async path also blocks localhost before any connection."""
        fetcher = LinkContentFetcher(retry_attempts=0)
        with pytest.raises(UnsafeFetchURLError):
            await fetcher.run_async(urls=["http://127.0.0.1:80/health"])


def _ssrf_hook_count(client) -> int:
    """Count SSRF guard request hooks installed on an httpx client."""
    guards = {_ssrf_guard_request_hook, _ssrf_guard_request_hook_async}
    return sum(1 for hook in client.event_hooks.get("request", []) if hook in guards)
