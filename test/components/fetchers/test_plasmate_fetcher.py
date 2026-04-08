# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PlasmateFetcher."""

import asyncio
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from haystack.components.fetchers.plasmate import PlasmateFetcher, _FORMAT_TO_MIME
from haystack.dataclasses import ByteStream


def _completed_process(stdout: str = "content", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = ""
    return m


class TestPlasmateFetcherInit:
    def test_defaults(self):
        fetcher = PlasmateFetcher.__new__(PlasmateFetcher)
        # Bypass _find_plasmate so the test doesn't require the binary
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            fetcher = PlasmateFetcher()
        assert fetcher.output_format == "markdown"
        assert fetcher.timeout == 30
        assert fetcher.selector is None
        assert fetcher.extra_headers == {}
        assert fetcher.raise_on_failure is True
        assert fetcher._mime_type == "text/markdown"

    def test_custom(self):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            fetcher = PlasmateFetcher(
                output_format="text",
                timeout=60,
                selector="main",
                extra_headers={"X-Custom": "val"},
                raise_on_failure=False,
            )
        assert fetcher.output_format == "text"
        assert fetcher.timeout == 60
        assert fetcher.selector == "main"
        assert fetcher.extra_headers == {"X-Custom": "val"}
        assert fetcher.raise_on_failure is False
        assert fetcher._mime_type == "text/plain"

    def test_mime_mapping(self):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            assert PlasmateFetcher(output_format="text")._mime_type == "text/plain"
            assert PlasmateFetcher(output_format="markdown")._mime_type == "text/markdown"
            assert PlasmateFetcher(output_format="som")._mime_type == "application/json"
            assert PlasmateFetcher(output_format="links")._mime_type == "text/plain"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="output_format must be one of"):
            PlasmateFetcher(output_format="html")

    def test_constructable_without_binary(self):
        """Component must be constructable even when the binary is missing,
        so pipelines can be serialized and loaded in other environments."""
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value=None,
        ):
            fetcher = PlasmateFetcher()
            assert fetcher._plasmate_bin is None


class TestPlasmateFetcherCmd:
    def _make(self, **kwargs):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            return PlasmateFetcher(**kwargs)

    def test_build_cmd_defaults(self):
        fetcher = self._make()
        cmd = fetcher._build_cmd("https://example.com")
        assert cmd[0] == "/usr/local/bin/plasmate"
        assert cmd[1] == "fetch"
        assert "https://example.com" in cmd
        assert "--format" in cmd
        assert "markdown" in cmd
        assert "--timeout" in cmd
        assert "30000" in cmd

    def test_build_cmd_with_selector(self):
        fetcher = self._make(selector="main")
        cmd = fetcher._build_cmd("https://example.com")
        assert "--selector" in cmd
        assert cmd[cmd.index("--selector") + 1] == "main"

    def test_build_cmd_with_headers(self):
        fetcher = self._make(extra_headers={"Authorization": "Bearer tok"})
        cmd = fetcher._build_cmd("https://example.com")
        assert "--header" in cmd
        assert "Authorization: Bearer tok" in cmd[cmd.index("--header") + 1]

    def test_build_cmd_timeout_converted_to_ms(self):
        fetcher = self._make(timeout=45)
        cmd = fetcher._build_cmd("https://example.com")
        assert "45000" in cmd

    def test_ensure_binary_raises_when_missing(self):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value=None,
        ):
            fetcher = PlasmateFetcher()
            with pytest.raises(ImportError, match="plasmate is required"):
                fetcher._ensure_binary()


class TestPlasmateFetcherRun:
    def _make(self, **kwargs):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            return PlasmateFetcher(**kwargs)

    def test_run_empty_urls(self):
        fetcher = self._make()
        result = fetcher.run(urls=[])
        assert result == {"streams": []}

    def test_run_single_url_success(self):
        fetcher = self._make(output_format="markdown")
        with patch(
            "subprocess.run",
            return_value=_completed_process("# Heading\n\nBody text"),
        ):
            result = fetcher.run(urls=["https://example.com"])

        streams = result["streams"]
        assert len(streams) == 1
        stream = streams[0]
        assert isinstance(stream, ByteStream)
        assert stream.data.decode() == "# Heading\n\nBody text"
        assert stream.meta["url"] == "https://example.com"
        assert stream.meta["content_type"] == "text/markdown"
        assert stream.mime_type == "text/markdown"

    def test_run_text_format(self):
        fetcher = self._make(output_format="text")
        with patch("subprocess.run", return_value=_completed_process("Plain text")):
            result = fetcher.run(urls=["https://example.com"])
        assert result["streams"][0].meta["content_type"] == "text/plain"

    def test_run_som_format(self):
        som = '{"role":"document","children":[{"role":"heading","name":"Title"}]}'
        fetcher = self._make(output_format="som")
        with patch("subprocess.run", return_value=_completed_process(som)):
            result = fetcher.run(urls=["https://example.com"])
        stream = result["streams"][0]
        assert stream.meta["content_type"] == "application/json"
        assert b"heading" in stream.data

    def test_run_single_url_raises_on_failure(self):
        fetcher = self._make(raise_on_failure=True)
        with patch(
            "subprocess.run",
            return_value=_completed_process("", returncode=1),
        ):
            with pytest.raises(RuntimeError, match="plasmate exited 1"):
                fetcher.run(urls=["https://example.com"])

    def test_run_single_url_suppresses_when_raise_false(self):
        fetcher = self._make(raise_on_failure=False)
        with patch(
            "subprocess.run",
            return_value=_completed_process("", returncode=1),
        ):
            result = fetcher.run(urls=["https://example.com"])
        assert len(result["streams"]) == 1
        assert result["streams"][0].data == b""

    def test_run_single_url_timeout(self):
        fetcher = self._make(raise_on_failure=False)
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="plasmate", timeout=30),
        ):
            result = fetcher.run(urls=["https://example.com"])
        assert result["streams"][0].data == b""

    def test_run_multiple_urls(self):
        fetcher = self._make()
        counter = {"n": 0}

        def fake_run(*args, **kwargs):
            counter["n"] += 1
            return _completed_process(f"content {counter['n']}")

        with patch("subprocess.run", side_effect=fake_run):
            result = fetcher.run(
                urls=["https://a.com", "https://b.com", "https://c.com"]
            )

        assert len(result["streams"]) == 3
        urls = {s.meta["url"] for s in result["streams"]}
        assert urls == {"https://a.com", "https://b.com", "https://c.com"}

    def test_run_missing_binary_raises_import_error(self):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value=None,
        ):
            fetcher = PlasmateFetcher()
            with patch(
                "subprocess.run",
                side_effect=FileNotFoundError("plasmate not found"),
            ):
                with pytest.raises(ImportError, match="plasmate is required"):
                    fetcher.run(urls=["https://example.com"])


class TestPlasmateFetcherAsync:
    def _make(self, **kwargs):
        with patch(
            "haystack.components.fetchers.plasmate._find_plasmate",
            return_value="/usr/local/bin/plasmate",
        ):
            return PlasmateFetcher(**kwargs)

    @pytest.mark.asyncio
    async def test_run_async_empty(self):
        fetcher = self._make()
        result = await fetcher.run_async(urls=[])
        assert result == {"streams": []}

    @pytest.mark.asyncio
    async def test_run_async_single(self):
        fetcher = self._make()
        with patch("subprocess.run", return_value=_completed_process("async content")):
            result = await fetcher.run_async(urls=["https://example.com"])
        assert len(result["streams"]) == 1
        assert result["streams"][0].data.decode() == "async content"

    @pytest.mark.asyncio
    async def test_run_async_concurrent(self):
        fetcher = self._make()
        counter = {"n": 0}

        def fake_run(*args, **kwargs):
            counter["n"] += 1
            return _completed_process(f"page {counter['n']}")

        with patch("subprocess.run", side_effect=fake_run):
            urls = [f"https://example.com/{i}" for i in range(5)]
            result = await fetcher.run_async(urls=urls)

        assert len(result["streams"]) == 5

    @pytest.mark.asyncio
    async def test_run_async_suppresses_errors_in_batch(self):
        fetcher = self._make(raise_on_failure=False)

        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return _completed_process("", returncode=1)
            return _completed_process(f"ok {call_count['n']}")

        with patch("subprocess.run", side_effect=fake_run):
            result = await fetcher.run_async(
                urls=["https://a.com", "https://b.com", "https://c.com"]
            )

        assert len(result["streams"]) == 3


class TestPlasmateFetcherIntegration:
    def test_format_mime_map_matches_valid_formats(self):
        from haystack.components.fetchers.plasmate import _VALID_FORMATS

        assert set(_VALID_FORMATS) == set(_FORMAT_TO_MIME.keys())

    def test_exported_from_package(self):
        from haystack.components.fetchers import PlasmateFetcher as Exported

        assert Exported is PlasmateFetcher
