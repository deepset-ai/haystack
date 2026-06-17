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
from urllib.parse import urlsplit

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

REQUEST_HEADERS = {
    "accept": "*/*",
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
    "referer": "https://www.google.com/",
}


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


class UnsafeFetchURLError(httpx.RequestError):
    """
    Raised when a URL is blocked by the Server-Side Request Forgery (SSRF) guard before a request is sent.

    Inherits from ``httpx.RequestError`` so it integrates with the existing retry/error-handling flow:
    ``raise_on_failure=False`` keeps logging and skipping the blocked URL, while ``raise_on_failure=True``
    surfaces the error to the caller.
    """


# Schemes considered safe to fetch over the network. Other schemes (e.g. ``file://``, ``ftp://``, ``gopher://``)
# can be abused to read local files or reach unexpected services and are rejected by the SSRF guard.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Determine whether an IP address points to a private, local, or otherwise non-routable network.

    This blocks loopback, private (RFC1918 / unique-local), link-local (including the
    cloud metadata address ``169.254.169.254``), reserved, multicast, and unspecified
    addresses for both IPv4 and IPv6. IPv4-mapped IPv6 addresses are unwrapped and
    re-checked so that, for example, ``::ffff:127.0.0.1`` is also blocked.

    :param ip: The IP address to inspect.
    :returns: ``True`` if the address must not be fetched, ``False`` otherwise.
    """
    # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:169.254.169.254) and re-check the embedded IPv4 address.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped

    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified


def _assert_safe_url(url: str | httpx.URL) -> None:
    """
    Validate that a URL is safe to fetch, raising :class:`UnsafeFetchURLError` if it is not.

    The check rejects unsupported schemes and any host that resolves (via DNS) to a
    private, loopback, link-local, or otherwise non-routable address. It is applied to
    the initial request URL and re-applied to every redirect hop, mitigating
    Server-Side Request Forgery (SSRF) attempts that target internal services or the
    cloud metadata endpoint, including DNS-rebinding and redirect-based bypasses.

    :param url: The URL about to be requested.
    :raises UnsafeFetchURLError: If the scheme is unsupported, the host is missing,
        the host cannot be resolved, or it resolves to a blocked address.
    """
    parts = urlsplit(str(url))
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise UnsafeFetchURLError(
            f"Blocked request to '{url}': URL scheme '{parts.scheme}' is not allowed. "
            f"Only {sorted(_ALLOWED_URL_SCHEMES)} URLs can be fetched."
        )

    host = parts.hostname
    if not host:
        raise UnsafeFetchURLError(f"Blocked request to '{url}': the URL does not contain a host.")

    # If the host is already an IP literal, validate it directly. Otherwise resolve all
    # addresses it maps to and reject the request if any of them is non-routable.
    try:
        literal_ip = ipaddress.ip_address(host)
        resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal_ip]
    except ValueError:
        try:
            addr_infos = socket.getaddrinfo(host, parts.port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeFetchURLError(f"Blocked request to '{url}': unable to resolve host '{host}'.") from exc
        resolved_ips = [ipaddress.ip_address(info[4][0]) for info in addr_infos]

    for ip in resolved_ips:
        if _is_blocked_ip(ip):
            raise UnsafeFetchURLError(
                f"Blocked request to '{url}': host '{host}' resolves to non-routable address '{ip}'. "
                "Fetching private, loopback, link-local, or cloud-metadata addresses is not allowed. "
                "Set `allow_private_addresses=True` on LinkContentFetcher to permit this (use with caution)."
            )


def _ssrf_guard_request_hook(request: httpx.Request) -> None:
    """
    httpx "request" event hook (sync client) that blocks SSRF-unsafe URLs before the request is sent.

    :param request: The outgoing httpx request (fired for the initial request and each redirect hop).
    :raises UnsafeFetchURLError: If the request target is not safe to fetch.
    """
    _assert_safe_url(request.url)


async def _ssrf_guard_request_hook_async(request: httpx.Request) -> None:
    """
    httpx "request" event hook (async client) that blocks SSRF-unsafe URLs before the request is sent.

    :param request: The outgoing httpx request (fired for the initial request and each redirect hop).
    :raises UnsafeFetchURLError: If the request target is not safe to fetch.
    """
    _assert_safe_url(request.url)


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
        allow_private_addresses: bool = False,
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
        :param allow_private_addresses: If `False` (the default), requests to hosts that resolve to private,
            loopback, link-local, or cloud-metadata addresses are blocked before any connection is made,
            and every redirect hop is re-validated. This mitigates Server-Side Request Forgery (SSRF) when
            the URLs come from an untrusted source (for example, when the component is exposed to an LLM as
            a tool). Set to `True` only if you explicitly trust the URLs and need to reach internal hosts.
        """
        self.raise_on_failure = raise_on_failure
        self.user_agents = user_agents or [DEFAULT_USER_AGENT]
        self.current_user_agent_idx: int = 0
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.http2 = http2
        self.client_kwargs = client_kwargs or {}
        self.request_headers = request_headers or {}
        self.allow_private_addresses = allow_private_addresses

        # Configure default client settings
        self.client_kwargs.setdefault("timeout", timeout)
        self.client_kwargs.setdefault("follow_redirects", True)

        # Create httpx clients
        client_kwargs = {**self.client_kwargs}

        # Optional HTTP/2 support
        if http2:
            try:
                h2_import.check()
                client_kwargs["http2"] = True
            except ImportError:
                logger.warning(
                    "HTTP/2 support requested but 'h2' package is not installed. "
                    "Falling back to HTTP/1.1. Install with `pip install httpx[http2]` to enable HTTP/2 support."
                )
                self.http2 = False  # Update the setting to match actual capability

        # Initialize synchronous client
        self._client = httpx.Client(**self._build_client_kwargs(client_kwargs, is_async=False))

        # Initialize asynchronous client
        self._async_client = httpx.AsyncClient(**self._build_client_kwargs(client_kwargs, is_async=True))

        # register default content handlers that extract data from the response
        self.handlers: dict[str, Callable[[httpx.Response], ByteStream]] = defaultdict(lambda: _text_content_handler)
        self.handlers["text/*"] = _text_content_handler
        self.handlers["text/html"] = _binary_content_handler
        self.handlers["application/json"] = _text_content_handler
        self.handlers["application/*"] = _binary_content_handler
        self.handlers["image/*"] = _binary_content_handler
        self.handlers["audio/*"] = _binary_content_handler
        self.handlers["video/*"] = _binary_content_handler

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=(retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError))),
            # This method is invoked only after failed requests (exception raised)
            after=self._switch_user_agent,
        )
        def get_response(url: str) -> httpx.Response:
            response = self._client.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response

        self._get_response: Callable = get_response

    def _build_client_kwargs(self, client_kwargs: dict[str, Any], is_async: bool) -> dict[str, Any]:
        """
        Return a copy of ``client_kwargs`` with the SSRF guard installed as a "request" event hook.

        Unless ``allow_private_addresses`` is set, the guard validates the initial URL and every redirect
        hop before the request leaves the process. httpx fires the "request" event hook for each
        (redirected) request, so this covers both the sync and async clients without changing the fetch
        logic. The sync client needs a regular callable while the async client needs a coroutine function,
        hence the two variants. Any user-supplied "request" hooks are preserved and run after the guard.

        :param client_kwargs: The base keyword arguments for the httpx client.
        :param is_async: Whether the kwargs are for an ``httpx.AsyncClient`` (``True``) or ``httpx.Client``.
        :returns: A new kwargs dict with the SSRF guard event hook merged in when enabled.
        """
        kwargs = {**client_kwargs}
        if self.allow_private_addresses:
            return kwargs

        guard = _ssrf_guard_request_hook_async if is_async else _ssrf_guard_request_hook
        event_hooks = dict(kwargs.get("event_hooks") or {})
        request_hooks = list(event_hooks.get("request") or [])
        kwargs["event_hooks"] = {**event_hooks, "request": [guard, *request_hooks]}
        return kwargs

    def _get_headers(self) -> dict[str, str]:
        """
        Build headers with precedence

        client defaults -> component defaults -> user-provided -> rotating UA
        """
        base = dict(self._client.headers)
        return _merge_headers(
            base, REQUEST_HEADERS, self.request_headers, {"User-Agent": self.user_agents[self.current_user_agent_idx]}
        )

    def __del__(self) -> None:
        """
        Clean up resources when the component is deleted.

        Closes both the synchronous and asynchronous HTTP clients to prevent
        resource leaks.
        """
        try:
            # Close the synchronous client if it exists
            if hasattr(self, "_client"):
                self._client.close()

            # There is no way to close the async client without await
        except Exception:
            # Suppress any exceptions during cleanup
            pass

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
        streams: list[ByteStream] = []
        if not urls:
            return {"streams": streams}

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

        finally:
            self.current_user_agent_idx = 0

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
        finally:
            self.current_user_agent_idx = 0

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

        while attempt <= self.retry_attempts:
            try:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exception = e
                attempt += 1
                if attempt <= self.retry_attempts:
                    self._switch_user_agent(None)  # Switch user agent for next retry
                    # Wait before retry using exponential backoff
                    await asyncio.sleep(min(2 * 2 ** (attempt - 1), 10))
                else:
                    break

        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception

        # This should never happen, but just in case
        raise httpx.RequestError("Failed to get response after retries", request=None)

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

    def _switch_user_agent(self, retry_state: RetryCallState | None = None) -> None:  # noqa: ARG002
        """
        Switches the User-Agent for this LinkContentRetriever to the next one in the list of user agents.

        Used by tenacity to retry the requests with a different user agent.

        :param retry_state: The retry state (unused, required by tenacity).
        """
        self.current_user_agent_idx = (self.current_user_agent_idx + 1) % len(self.user_agents)
        logger.debug("Switched user agent to {user_agent}", user_agent=self.user_agents[self.current_user_agent_idx])
