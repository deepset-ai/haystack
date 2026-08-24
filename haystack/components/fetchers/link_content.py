# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import ipaddress
import socket
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fnmatch import fnmatch
from typing import Any, cast

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from haystack import component, logging
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.version import __version__

# HTTP/2 support via lazy import
with LazyImport("Run 'pip install httpx[http2]' to use HTTP/2 support") as h2_import:
    pass  # nothing to import as we simply set the http2 attribute, library handles the rest

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = f"haystack/LinkContentFetcher/{__version__}"

DEFAULT_MAX_RESPONSE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 5

# IPv4 shared address space (RFC 6598), commonly used for internal networks and VPN overlays like Tailscale.
# `ipaddress` does not classify it as private, so it's checked explicitly.
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")

REQUEST_HEADERS = {
    "accept": "*/*",
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
    "referer": "https://www.google.com/",
}


class UnsafeTargetError(ValueError):
    """
    Raised when a URL or one of its redirect targets points to a host that must not be fetched from.

    This includes hosts outside the ``allowed_hosts`` whitelist and hosts that resolve to a private, loopback,
    link-local, multicast, or otherwise internal IP address.
    """


class ResponseTooLargeError(ValueError):
    """
    Raised when a response body exceeds the fetcher's ``max_response_bytes`` limit.
    """


def _resolve_host(host: str, port: int) -> list[str]:
    """
    Resolves a host name to all of its IPv4/IPv6 addresses (A/AAAA records).

    :param host: The host name or IP literal to resolve.
    :param port: The port to resolve for. It does not affect the returned addresses.
    :returns: The resolved addresses as strings. An empty list if resolution fails, so that the underlying
        HTTP client can surface its own (retryable) connection error.
    """
    try:
        addr_infos = socket.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError):
        return []
    return [addr_info[4][0] for addr_info in addr_infos]


def _is_forbidden_ip(address: str) -> bool:
    """
    Checks whether an IP address belongs to a range the fetcher must never connect to.

    This covers private, loopback, link-local, multicast, reserved, unspecified, and unique-local addresses, as
    well as the RFC 6598 shared address space. Unparseable addresses are rejected as well.

    :param address: The IP address to check, as a string.
    :returns: `True` if the address is forbidden.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip in _SHARED_ADDRESS_SPACE
    )


def _host_matches_allowlist(host: str, allowed_hosts: list[str]) -> bool:
    """
    Checks whether a host matches any entry of a domain suffix whitelist.

    A host matches an entry if it is equal to it or a subdomain of it. For example, `api.example.com` matches
    `example.com` but `notexample.com` does not. Comparison is case-insensitive and ignores a trailing dot
    (the DNS root).

    :param host: The host name to check.
    :param allowed_hosts: The whitelist entries.
    :returns: `True` if the host matches at least one entry.
    """
    normalized_host = host.lower().rstrip(".")
    for allowed_host in allowed_hosts:
        normalized_allowed = allowed_host.lower().rstrip(".")
        if normalized_host == normalized_allowed or normalized_host.endswith(f".{normalized_allowed}"):
            return True
    return False


def _merge_headers(*args: dict[str, str]) -> dict[str, str]:
    """
    Merge a list of dict using case-insensitively

    :param args: a list of dict to merge
    :returns: The merged dict
    """
    merged = {}
    keymap = {}

    for d in args:
        for k, v in d.items():
            kl = k.lower()
            keymap[kl] = k
            merged[kl] = v

    return {keymap[kl]: v for kl, v in merged.items()}


def _text_content_handler(response: httpx.Response) -> ByteStream:
    """
    Handles text content.

    :param response: Response object from the request.
    :returns: The extracted text.
    """
    return ByteStream.from_string(response.text)


def _binary_content_handler(response: httpx.Response) -> ByteStream:
    """
    Handles binary content.

    :param response: Response object from the request.
    :returns: The extracted binary file-like object.
    """
    return ByteStream(data=response.content)


@component
class LinkContentFetcher:
    """
    Fetches and extracts content from URLs.

    It supports various content types, retries on failures, and automatic user-agent rotation for failed web
    requests. Use it as the data-fetching step in your pipelines.

    For security, every request target is validated before the request is made, including each redirect hop:
    hosts can be restricted to an `allowed_hosts` domain suffix whitelist, and hosts resolving to private,
    loopback, link-local, multicast, or otherwise internal IP addresses are rejected. Response bodies are
    streamed and capped at `max_response_bytes`.

    You may need to convert LinkContentFetcher's output into a list of documents. Use HTMLToDocument
    converter to do this.

    ### Usage example

    ```python
    from haystack.components.fetchers.link_content import LinkContentFetcher

    fetcher = LinkContentFetcher()
    streams = fetcher.run(urls=["https://www.google.com"])["streams"]

    assert len(streams) == 1
    assert streams[0].meta == {'content_type': 'text/html', 'url': 'https://www.google.com'}
    assert streams[0].data
    ```

    For async usage:

    ```python
    import asyncio
    from haystack.components.fetchers import LinkContentFetcher

    async def fetch_async():
        fetcher = LinkContentFetcher()
        result = await fetcher.run_async(urls=["https://www.google.com"])
        return result["streams"]

    streams = asyncio.run(fetch_async())
    ```
    """

    def __init__(
        self,
        raise_on_failure: bool = True,
        user_agents: list[str] | None = None,
        retry_attempts: int = 2,
        timeout: int = 3,
        http2: bool = False,
        client_kwargs: dict | None = None,
        request_headers: dict[str, str] | None = None,
        allowed_hosts: list[str] | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
    ) -> None:
        """
        Initializes the component.

        :param raise_on_failure: If `True`, raises an exception if it fails to fetch a single URL.
            For multiple URLs, it logs errors and returns the content it successfully fetched.
        :param user_agents: [User agents](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent)
            for fetching content. If `None`, a default user agent is used.
        :param retry_attempts: The number of times to retry to fetch the URL's content.
        :param timeout: Timeout in seconds for the request.
        :param http2: Whether to enable HTTP/2 support for requests. Defaults to False.
                     Requires the 'h2' package to be installed (via `pip install httpx[http2]`).
        :param client_kwargs: Additional keyword arguments to pass to the httpx client.
                     If `None`, default values are used.
                     `follow_redirects` is always overridden to `False`: redirects are followed manually,
                     one hop at a time, so that every hop can be validated. To disable redirect following
                     entirely, pass `{"follow_redirects": False}`.
        :param request_headers: Additional headers to send with every request. These take precedence over the
                     component's default headers but not over the rotating `User-Agent`.
        :param allowed_hosts: Optional whitelist of allowed domain suffixes, for example
                     `["example.com", "cdn.example.org"]`. A host is allowed if it equals one of the entries
                     or is a subdomain of one (so `api.example.com` matches `example.com`, but
                     `notexample.com` does not). If `None`, any host is allowed. Hosts are additionally
                     always checked against forbidden IP ranges, regardless of this whitelist.
        :param max_response_bytes: Maximum size in bytes of a response body. Responses are streamed and
                     bodies exceeding this limit raise a `ResponseTooLargeError`.
        :param max_redirects: Maximum number of redirects to follow. Each hop is validated before the
                     request is made.
        """
        self.raise_on_failure = raise_on_failure
        self.user_agents = user_agents or [DEFAULT_USER_AGENT]
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.http2 = http2
        self.client_kwargs = client_kwargs or {}
        self.request_headers = request_headers or {}
        self.allowed_hosts = allowed_hosts
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects

        # Configure default client settings
        self.client_kwargs.setdefault("timeout", timeout)
        # Redirects are followed manually (see `_request_following_redirects`) so that every hop can be
        # validated before the request is made; the underlying client has `follow_redirects` forced to `False`
        # in `_build_client_kwargs`. Here we only remember whether the user wants redirects followed at all.
        self._follow_redirects = self.client_kwargs.get("follow_redirects", True)

        # httpx clients are built lazily in warm_up / warm_up_async (resource lifecycle)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

        # register default content handlers that extract data from the response
        self.handlers: dict[str, Callable[[httpx.Response], ByteStream]] = defaultdict(lambda: _text_content_handler)
        self.handlers["text/*"] = _text_content_handler
        self.handlers["text/html"] = _binary_content_handler
        self.handlers["application/json"] = _text_content_handler
        self.handlers["application/*"] = _binary_content_handler
        self.handlers["image/*"] = _binary_content_handler
        self.handlers["audio/*"] = _binary_content_handler
        self.handlers["video/*"] = _binary_content_handler

    def _get_response(self, url: str) -> httpx.Response:
        """
        Gets a response from a URL, rotating the user agent on every failed attempt.

        The rotation cursor is local to this call: `run` fetches URLs concurrently, so a cursor kept on the
        component would be advanced and reset by whichever fetches happen to be in flight at the same time.

        :param url: The URL to fetch.
        :returns: The httpx Response object.
        """
        user_agent_idx = 0

        def rotate_user_agent(retry_state: RetryCallState) -> None:  # noqa: ARG001
            nonlocal user_agent_idx
            user_agent_idx = (user_agent_idx + 1) % len(self.user_agents)
            logger.debug("Switched user agent to {user_agent}", user_agent=self.user_agents[user_agent_idx])

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts + 1),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=(retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError))),
            # This callback is invoked only after failed requests (exception raised)
            after=rotate_user_agent,
        )
        def get_response(url: str) -> httpx.Response:
            assert self._client is not None  # mypy: client is built by warm_up before run
            headers = self._get_headers(self.user_agents[user_agent_idx])
            return self._request_following_redirects(self._client, url, headers)

        return get_response(url)

    def _validate_target(self, url: str) -> None:
        """
        Validates that a URL may be fetched before the request is made.

        The URL's host must match the `allowed_hosts` whitelist (if set) and must not resolve to any forbidden
        IP address (private, loopback, link-local, multicast, reserved, unspecified, unique-local, or shared
        address space ranges). Note that this mitigates, but cannot fully eliminate, DNS rebinding: the HTTP
        client performs its own name resolution when connecting.

        :param url: The URL to validate.
        :raises UnsafeTargetError: If the URL's host is not allowed or resolves to a forbidden address.
        """
        parsed = httpx.URL(url)
        host = parsed.host
        if not host:
            raise UnsafeTargetError(f"URL '{url}' has no host")

        if self.allowed_hosts is not None and not _host_matches_allowlist(host, self.allowed_hosts):
            raise UnsafeTargetError(
                f"Host '{host}' (URL '{url}') is not in the allowed_hosts whitelist: {self.allowed_hosts}"
            )

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        for address in _resolve_host(host, port):
            if _is_forbidden_ip(address):
                raise UnsafeTargetError(
                    f"Host '{host}' (URL '{url}') resolves to the forbidden IP address '{address}'. "
                    "Requests to private, loopback, link-local, multicast, or other internal addresses "
                    "are not allowed."
                )

    def _request_following_redirects(self, client: httpx.Client, url: str, headers: dict[str, str]) -> httpx.Response:
        """
        Performs the request for a URL, following redirects manually one hop at a time.

        The HTTP client itself never follows redirects (see `_build_client_kwargs`): every hop, starting from
        the original URL, is validated before its request is made. This prevents a redirect from a trusted host
        to an internal address from ever being fetched.

        :param client: The httpx client to make the requests with.
        :param url: The URL to fetch.
        :param headers: Headers to send with every hop.
        :returns: The final httpx Response object, with its body already read.
        :raises httpx.TooManyRedirects: If more than `max_redirects` redirects are encountered.
        """
        current_url = url
        for _hop in range(self.max_redirects + 1):
            self._validate_target(current_url)
            response = self._request_hop(client, current_url, headers)
            if not (self._follow_redirects and response.is_redirect):
                response.raise_for_status()
                return response
            current_url = str(httpx.URL(current_url).join(response.headers["location"]))
        raise httpx.TooManyRedirects(f"Exceeded {self.max_redirects} redirects.", request=httpx.Request("GET", url))

    def _request_hop(self, client: httpx.Client, url: str, headers: dict[str, str]) -> httpx.Response:
        """
        Performs a single GET request, streaming the response body with a size cap.

        The body is read while the connection is still open so that the transfer is aborted as soon as it
        exceeds `max_response_bytes`. A plain, fully-read response is returned so that content handlers can
        access it like a regular, non-streaming response.

        Redirect responses are returned without reading their body: it is never used.

        :param client: The httpx client to make the request with.
        :param url: The URL to fetch.
        :param headers: Headers to send with the request.
        :returns: The httpx Response object for this hop, with its body read unless it is a redirect.
        :raises ResponseTooLargeError: If the response body exceeds `max_response_bytes`.
        """
        with client.stream("GET", url, headers=headers) as response:
            body = b""
            if not (self._follow_redirects and response.is_redirect):
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > self.max_response_bytes:
                        raise ResponseTooLargeError(
                            f"Response from '{url}' exceeds max_response_bytes={self.max_response_bytes} "
                            f"(received at least {len(chunks)} bytes)."
                        )
                body = bytes(chunks)
            # Rebuild a fully-read response so handlers can use `response.text`/`response.content` as usual.
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                request=response.request,
                default_encoding=response.default_encoding,
            )

    def _build_client_kwargs(self) -> dict[str, Any]:
        """
        Build the keyword arguments used to construct the httpx clients.

        Resolves optional HTTP/2 support, downgrading to HTTP/1.1 if the 'h2' package is not installed.
        """
        client_kwargs = {**self.client_kwargs}

        # Redirects are followed manually, one validated hop at a time (see `_request_following_redirects`),
        # so the underlying client must never follow them on its own.
        client_kwargs["follow_redirects"] = False

        # Optional HTTP/2 support
        if self.http2:
            try:
                h2_import.check()
                client_kwargs["http2"] = True
            except ImportError:
                logger.warning(
                    "HTTP/2 support requested but 'h2' package is not installed. "
                    "Falling back to HTTP/1.1. Install with `pip install httpx[http2]` to enable HTTP/2 support."
                )
                self.http2 = False  # Update the setting to match actual capability

        return client_kwargs

    def warm_up(self) -> None:
        """
        Initializes the synchronous httpx client.
        """
        if self._client is None:
            self._client = httpx.Client(**self._build_client_kwargs())

    async def warm_up_async(self) -> None:  # noqa: RUF029
        """
        Initializes the asynchronous httpx client on the serving event loop.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._build_client_kwargs())

    def close(self) -> None:
        """
        Releases the synchronous httpx client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Releases the asynchronous httpx client.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def _get_headers(self, user_agent: str) -> dict[str, str]:
        """
        Build headers with precedence

        client defaults -> component defaults -> user-provided -> rotating UA

        :param user_agent: The user agent for this attempt, taken from the caller's own rotation.
        """
        base = dict(self._client.headers) if self._client is not None else {}
        return _merge_headers(base, REQUEST_HEADERS, self.request_headers, {"User-Agent": user_agent})

    @component.output_types(streams=list[ByteStream])
    def run(self, urls: list[str]) -> dict[str, Any]:
        """
        Fetches content from a list of URLs and returns a list of extracted content streams.

        Each content stream is a `ByteStream` object containing the extracted content as binary data.
        Each ByteStream object in the returned list corresponds to the contents of a single URL.
        The content type of each stream is stored in the metadata of the ByteStream object under
        the key "content_type". The URL of the fetched content is stored under the key "url".

        :param urls: A list of URLs to fetch content from.
        :returns: `ByteStream` objects representing the extracted content.

        :raises Exception: If the provided list of URLs contains only a single URL, and `raise_on_failure` is set to
            `True`, an exception will be raised in case of an error during content retrieval.
            In all other scenarios, any retrieval errors are logged, and a list of successfully retrieved `ByteStream`
             objects is returned.
        """
        self.warm_up()

        streams: list[ByteStream] = []
        if not urls:
            return {"streams": streams}

        # don't use multithreading if there's only one URL
        if len(urls) == 1:
            stream_metadata, stream = self._fetch(urls[0])
            stream.meta.update(stream_metadata)
            stream = replace(stream, mime_type=stream.meta.get("content_type", None))
            streams.append(stream)
        else:
            with ThreadPoolExecutor() as executor:
                results = executor.map(self._fetch_with_exception_suppression, urls)

            for stream_metadata, stream in results:  # type: ignore
                if stream_metadata is not None and stream is not None:
                    stream.meta.update(stream_metadata)
                    stream = replace(stream, mime_type=stream.meta.get("content_type", None))
                    streams.append(stream)

        return {"streams": streams}

    @component.output_types(streams=list[ByteStream])
    async def run_async(self, urls: list[str]) -> dict[str, Any]:
        """
        Asynchronously fetches content from a list of URLs and returns a list of extracted content streams.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param urls: A list of URLs to fetch content from.
        :returns: `ByteStream` objects representing the extracted content.
        """
        await self.warm_up_async()

        streams: list[ByteStream] = []
        if not urls:
            return {"streams": streams}

        assert self._async_client is not None  # mypy: async_client is built by warm_up_async above
        # Create tasks for all URLs using _fetch_async directly
        tasks = [self._fetch_async(url, self._async_client) for url in urls]

        # Only capture exceptions when we have multiple URLs or raise_on_failure=False
        # This ensures errors propagate appropriately for single URLs with raise_on_failure=True
        return_exceptions = not (len(urls) == 1 and self.raise_on_failure)
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Process results
        for i, result in enumerate(results):
            # Handle exception results (only happens when return_exceptions=True)
            if isinstance(result, Exception):
                logger.warning("Error fetching {url}: {error}", url=urls[i], error=str(result))
                # Add an empty result for failed URLs when raise_on_failure=False
                if not self.raise_on_failure:
                    streams.append(ByteStream(data=b"", meta={"content_type": "Unknown", "url": urls[i]}))
                continue

            # Process successful results
            # At this point, result is not an exception, so we need to cast it to the correct type for mypy
            if not isinstance(result, Exception):  # Runtime check
                # Use cast to tell mypy that result is the tuple type returned by _fetch_async
                result_tuple = cast(tuple[dict[str, str] | None, ByteStream | None], result)
                stream_metadata, stream = result_tuple
                if stream_metadata is not None and stream is not None:
                    stream.meta.update(stream_metadata)
                    stream = replace(stream, mime_type=stream.meta.get("content_type", None))
                    streams.append(stream)

        return {"streams": streams}

    def _fetch(self, url: str) -> tuple[dict[str, str], ByteStream]:
        """
        Fetches content from a URL and returns it as a ByteStream.

        :param url: The URL to fetch content from.
        :returns: A tuple containing the ByteStream metadata dict and the corresponding ByteStream.
             ByteStream metadata contains the URL and the content type of the fetched content.
             The content type is a string indicating the type of content fetched (for example, "text/html",
             "application/pdf"). The ByteStream object contains the fetched content as binary data.

        :raises: If an error occurs during content retrieval and `raise_on_failure` is set to True, this method will
        raise an exception. Otherwise, all fetching errors are logged, and an empty ByteStream is returned.

        """
        content_type: str = "text/html"
        stream: ByteStream = ByteStream(data=b"")
        try:
            response = self._get_response(url)
            content_type = self._get_content_type(response)
            handler: Callable = self._resolve_handler(content_type)
            stream = handler(response)
        except Exception as e:
            if self.raise_on_failure:
                raise e
            # less verbose log as this is expected to happen often (requests failing, blocked, etc.)
            logger.debug("Couldn't retrieve content from {url} because {error}", url=url, error=str(e))

        return {"content_type": content_type, "url": url}, stream

    async def _fetch_async(
        self, url: str, client: httpx.AsyncClient
    ) -> tuple[dict[str, str] | None, ByteStream | None]:
        """
        Asynchronously fetches content from a URL and returns it as a ByteStream.

        :param url: The URL to fetch content from.
        :param client: The async httpx client to use for making requests.
        :returns: A tuple containing the ByteStream metadata dict and the corresponding ByteStream.
        """
        content_type: str = "text/html"
        stream: ByteStream | None = None
        metadata: dict[str, str] | None = None

        try:
            response = await self._get_response_async(url, client)
            content_type = self._get_content_type(response)
            handler: Callable = self._resolve_handler(content_type)
            stream = handler(response)
            metadata = {"content_type": content_type, "url": url}
        except Exception as e:
            if self.raise_on_failure:
                raise e
            logger.debug("Couldn't retrieve content from {url} because {error}", url=url, error=str(e))
            # Create an empty ByteStream for failed requests when raise_on_failure is False
            stream = ByteStream(data=b"")
            metadata = {"content_type": content_type, "url": url}

        return metadata, stream

    def _fetch_with_exception_suppression(self, url: str) -> tuple[dict[str, str] | None, ByteStream | None]:
        """
        Fetches content from a URL and returns it as a ByteStream.

        If `raise_on_failure` is set to True, this method will wrap the fetch() method and catch any exceptions.
        Otherwise, it will simply call the fetch() method.
        :param url: The URL to fetch content from.
        :returns: A tuple containing the ByteStream metadata dict and the corresponding ByteStream.

        """
        if self.raise_on_failure:
            try:
                return self._fetch(url)
            except Exception as e:
                logger.warning("Error fetching {url}: {error}", url=url, error=str(e))
                return {"content_type": "Unknown", "url": url}, None
        else:
            return self._fetch(url)

    async def _get_response_async(self, url: str, client: httpx.AsyncClient) -> httpx.Response:
        """
        Asynchronously gets a response from a URL with retry logic.

        :param url: The URL to fetch.
        :param client: The async httpx client to use for making requests.
        :returns: The httpx Response object.
        """
        attempt = 0
        last_exception = None
        # Local to this call: `run_async` gathers URLs concurrently, so a cursor on the component
        # would be shared by every request in flight.
        user_agent_idx = 0

        while attempt <= self.retry_attempts:
            try:
                headers = self._get_headers(self.user_agents[user_agent_idx])
                response = await self._request_following_redirects_async(client, url, headers)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                attempt += 1
                if attempt <= self.retry_attempts:
                    # Switch user agent for next retry
                    user_agent_idx = (user_agent_idx + 1) % len(self.user_agents)
                    # Wait before retry using exponential backoff
                    await asyncio.sleep(min(2 * 2 ** (attempt - 1), 10))
                else:
                    break

        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception

        # This should never happen, but just in case
        raise httpx.RequestError("Failed to get response after retries", request=None)

    async def _request_following_redirects_async(
        self, client: httpx.AsyncClient, url: str, headers: dict[str, str]
    ) -> httpx.Response:
        """
        Asynchronously performs the request for a URL, following redirects manually one hop at a time.

        Every hop, starting from the original URL, is validated before its request is made. This is the
        asynchronous counterpart of `_request_following_redirects`.

        :param client: The async httpx client to make the requests with.
        :param url: The URL to fetch.
        :param headers: Headers to send with every hop.
        :returns: The final httpx Response object, with its body already read.
        :raises httpx.TooManyRedirects: If more than `max_redirects` redirects are encountered.
        """
        current_url = url
        for _hop in range(self.max_redirects + 1):
            self._validate_target(current_url)
            response = await self._request_hop_async(client, current_url, headers)
            if not (self._follow_redirects and response.is_redirect):
                response.raise_for_status()
                return response
            current_url = str(httpx.URL(current_url).join(response.headers["location"]))
        raise httpx.TooManyRedirects(f"Exceeded {self.max_redirects} redirects.", request=httpx.Request("GET", url))

    async def _request_hop_async(self, client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> httpx.Response:
        """
        Asynchronously performs a single GET request, streaming the response body with a size cap.

        This is the asynchronous counterpart of `_request_hop`.

        :param client: The async httpx client to make the request with.
        :param url: The URL to fetch.
        :param headers: Headers to send with the request.
        :returns: The httpx Response object for this hop, with its body read unless it is a redirect.
        :raises ResponseTooLargeError: If the response body exceeds `max_response_bytes`.
        """
        async with client.stream("GET", url, headers=headers) as response:
            body = b""
            if not (self._follow_redirects and response.is_redirect):
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > self.max_response_bytes:
                        raise ResponseTooLargeError(
                            f"Response from '{url}' exceeds max_response_bytes={self.max_response_bytes} "
                            f"(received at least {len(chunks)} bytes)."
                        )
                body = bytes(chunks)
            # Rebuild a fully-read response so handlers can use `response.text`/`response.content` as usual.
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                request=response.request,
                default_encoding=response.default_encoding,
            )

    def _get_content_type(self, response: httpx.Response) -> str:
        """
        Get the content type of the response.

        :param response: The response object.
        :returns: The content type of the response.
        """
        content_type = response.headers.get("Content-Type", "")
        return content_type.split(";")[0]

    def _resolve_handler(self, content_type: str) -> Callable[[httpx.Response], ByteStream]:
        """
        Resolves the handler for the given content type.

        First, it tries to find a direct match for the content type in the handlers dictionary.
        If no direct match is found, it tries to find a pattern match using the fnmatch function.
        If no pattern match is found, it returns the default handler for text/plain.

        :param content_type: The content type to resolve the handler for.
        :returns: The handler for the given content type, if found. Otherwise, the default handler for text/plain.
        """
        # direct match
        if content_type in self.handlers:
            return self.handlers[content_type]

        # pattern matches
        for pattern, handler in self.handlers.items():
            if fnmatch(content_type, pattern):
                return handler

        # default handler
        return self.handlers["text/plain"]
