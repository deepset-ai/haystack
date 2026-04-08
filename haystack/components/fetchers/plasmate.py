# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
PlasmateFetcher — lightweight alternative to LinkContentFetcher.

Uses Plasmate (https://github.com/plasmate-labs/plasmate) instead of Chrome or
raw HTTP + HTML parsing. Plasmate is an open-source Rust browser engine that
returns Structured Object Model (SOM), markdown, or plain text rather than raw
HTML. Typical output is 10-100x smaller, meaning downstream LLM calls consume
far fewer tokens.

Install: ``pip install plasmate``
Docs:    https://plasmate.app
"""

import asyncio
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from haystack import component, logging
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

_VALID_FORMATS = ("text", "markdown", "som", "links")

_INSTALL_MSG = (
    "plasmate is required for PlasmateFetcher. "
    "Install it with: pip install plasmate\n"
    "Docs: https://plasmate.app"
)

_FORMAT_TO_MIME = {
    "text": "text/plain",
    "markdown": "text/markdown",
    "som": "application/json",
    "links": "text/plain",
}


def _find_plasmate() -> str | None:
    """Return the resolved path to the plasmate binary, or None."""
    path = shutil.which("plasmate")
    if path:
        return path
    try:
        import plasmate as _p  # noqa: F401
        return shutil.which("plasmate")
    except ImportError:
        return None


@component
class PlasmateFetcher:
    """
    Fetches content from URLs using Plasmate — a lightweight alternative to Chrome.

    PlasmateFetcher is a drop-in replacement for
    :class:`~haystack.components.fetchers.LinkContentFetcher` on static and
    server-rendered pages. It uses the Plasmate Rust browser engine instead of
    spawning a browser or parsing raw HTML, returning pre-processed content
    (plain text, markdown, or SOM) that is typically 10-100x smaller than raw
    HTML. This directly reduces token consumption in downstream LLM components.

    ### Usage example

    ```python
    from haystack.components.fetchers import PlasmateFetcher

    fetcher = PlasmateFetcher(output_format="markdown")
    streams = fetcher.run(urls=["https://docs.haystack.deepset.ai/"])["streams"]

    assert len(streams) == 1
    assert streams[0].meta["content_type"] == "text/markdown"
    assert streams[0].data  # already a compact, LLM-ready string
    ```

    Async usage:

    ```python
    import asyncio
    from haystack.components.fetchers import PlasmateFetcher

    async def fetch_async():
        fetcher = PlasmateFetcher()
        result = await fetcher.run_async(urls=["https://example.com"])
        return result["streams"]

    streams = asyncio.run(fetch_async())
    ```

    In a pipeline:

    ```python
    from haystack import Pipeline
    from haystack.components.fetchers import PlasmateFetcher
    from haystack.components.converters import TextFileToDocument

    pipe = Pipeline()
    pipe.add_component("fetcher", PlasmateFetcher(output_format="markdown"))
    pipe.add_component("converter", TextFileToDocument())
    pipe.connect("fetcher.streams", "converter.sources")

    result = pipe.run({"fetcher": {"urls": ["https://example.com"]}})
    ```
    """

    def __init__(
        self,
        output_format: str = "markdown",
        timeout: int = 30,
        selector: str | None = None,
        extra_headers: dict[str, str] | None = None,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Initializes the component.

        :param output_format: The format Plasmate should emit. One of
            ``"text"``, ``"markdown"`` (default), ``"som"`` (full JSON), or
            ``"links"``. Markdown is the best default for most RAG pipelines.
        :param timeout: Per-request timeout in seconds. Defaults to 30.
        :param selector: Optional ARIA role or CSS id selector to scope
            extraction (e.g. ``"main"`` or ``"#article"``).
        :param extra_headers: Optional HTTP headers forwarded with each request.
        :param raise_on_failure: If ``True``, raises an exception when fetching
            a single URL fails. For multiple URLs, errors are logged and
            successfully fetched streams are returned.
        """
        if output_format not in _VALID_FORMATS:
            raise ValueError(
                f"output_format must be one of {_VALID_FORMATS}; got {output_format!r}"
            )
        self.output_format = output_format
        self.timeout = timeout
        self.selector = selector
        self.extra_headers = extra_headers or {}
        self.raise_on_failure = raise_on_failure
        self._mime_type = _FORMAT_TO_MIME[output_format]

        # Resolve the binary once at init time. If missing we still allow the
        # component to be constructed (so pipelines can be serialized), but the
        # first run() call will raise a clear error.
        self._plasmate_bin: str | None = _find_plasmate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_binary(self) -> str:
        """Return the plasmate binary path, raising a clear error if missing."""
        if self._plasmate_bin is None:
            self._plasmate_bin = _find_plasmate()
        if self._plasmate_bin is None:
            raise ImportError(_INSTALL_MSG)
        return self._plasmate_bin

    def _build_cmd(self, url: str) -> list[str]:
        """Build the plasmate CLI command for a URL."""
        cmd = [
            self._ensure_binary(),
            "fetch",
            url,
            "--format", self.output_format,
            "--timeout", str(self.timeout * 1000),  # plasmate uses ms
        ]
        if self.selector:
            cmd += ["--selector", self.selector]
        for key, value in self.extra_headers.items():
            cmd += ["--header", f"{key}: {value}"]
        return cmd

    def _fetch(self, url: str) -> tuple[dict[str, str], ByteStream]:
        """Synchronously run plasmate for a single URL."""
        try:
            result = subprocess.run(
                self._build_cmd(url),
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            if result.returncode != 0:
                msg = f"plasmate exited {result.returncode}: {result.stderr[:200]}"
                if self.raise_on_failure:
                    raise RuntimeError(msg)
                logger.debug("Couldn't retrieve content from {url}: {error}", url=url, error=msg)
                return (
                    {"content_type": self._mime_type, "url": url},
                    ByteStream(data=b""),
                )
            content = result.stdout
            stream = ByteStream.from_string(content)
            return {"content_type": self._mime_type, "url": url}, stream
        except subprocess.TimeoutExpired as e:
            if self.raise_on_failure:
                raise
            logger.debug("Timeout fetching {url}: {error}", url=url, error=str(e))
            return (
                {"content_type": self._mime_type, "url": url},
                ByteStream(data=b""),
            )
        except FileNotFoundError as e:
            raise ImportError(_INSTALL_MSG) from e

    async def _fetch_async(self, url: str) -> tuple[dict[str, str], ByteStream]:
        """Async wrapper — runs _fetch in the default thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch, url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @component.output_types(streams=list[ByteStream])
    def run(self, urls: list[str]) -> dict[str, Any]:
        """
        Fetches content from a list of URLs and returns extracted content streams.

        Each ``ByteStream`` contains pre-processed content (text, markdown, or
        SOM JSON) rather than raw HTML. ``meta`` includes ``content_type`` and
        ``url``.

        :param urls: A list of URLs to fetch content from.
        :returns: A dictionary with a ``streams`` key containing the list of
            fetched :class:`~haystack.dataclasses.ByteStream` objects.

        :raises ImportError: If the plasmate binary is not installed.
        :raises RuntimeError: If a fetch fails and ``raise_on_failure`` is
            ``True`` and only a single URL was provided.
        """
        streams: list[ByteStream] = []
        if not urls:
            return {"streams": streams}

        if len(urls) == 1:
            metadata, stream = self._fetch(urls[0])
            stream.meta.update(metadata)
            stream = replace(stream, mime_type=stream.meta.get("content_type"))
            streams.append(stream)
            return {"streams": streams}

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._fetch_with_suppression, urls))

        for metadata, stream in results:
            if metadata is None or stream is None:
                continue
            stream.meta.update(metadata)
            stream = replace(stream, mime_type=stream.meta.get("content_type"))
            streams.append(stream)

        return {"streams": streams}

    @component.output_types(streams=list[ByteStream])
    async def run_async(self, urls: list[str]) -> dict[str, Any]:
        """
        Asynchronously fetches content from a list of URLs.

        Async version of :meth:`run` with identical semantics.

        :param urls: A list of URLs to fetch content from.
        :returns: A dictionary with a ``streams`` key containing the list of
            fetched :class:`~haystack.dataclasses.ByteStream` objects.
        """
        streams: list[ByteStream] = []
        if not urls:
            return {"streams": streams}

        return_exceptions = not (len(urls) == 1 and self.raise_on_failure)
        results = await asyncio.gather(
            *[self._fetch_async(url) for url in urls],
            return_exceptions=return_exceptions,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Error fetching {url}: {error}", url=urls[i], error=str(result))
                if not self.raise_on_failure:
                    streams.append(
                        ByteStream(
                            data=b"",
                            meta={"content_type": self._mime_type, "url": urls[i]},
                        )
                    )
                continue

            metadata, stream = result  # type: ignore[assignment]
            stream.meta.update(metadata)
            stream = replace(stream, mime_type=stream.meta.get("content_type"))
            streams.append(stream)

        return {"streams": streams}

    def _fetch_with_suppression(self, url: str) -> tuple[dict[str, str] | None, ByteStream | None]:
        """Fetch a URL, logging rather than raising when raise_on_failure is True for batches."""
        try:
            return self._fetch(url)
        except Exception as e:  # noqa: BLE001 — batch mode suppresses per-URL errors
            logger.warning("Error fetching {url}: {error}", url=url, error=str(e))
            return {"content_type": "Unknown", "url": url}, None
