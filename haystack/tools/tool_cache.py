# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Opt-in keyed cache for tool invocations.

Used by ToolInvoker and Agent to avoid re-running identical tool calls within an agent loop.
See https://github.com/deepset-ai/haystack/issues/11588 for the motivating problem
statement and design discussion.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any

from haystack import logging

logger = logging.getLogger(__name__)


def _canonicalize_args(arguments: dict[str, Any]) -> str:
    """
    Produce a stable JSON string for a tool-call arguments dict, used as part of the cache key.

    Falls back to `str(arguments)` if the arguments aren't JSON-serializable (e.g. contain custom
    objects), so caching degrades gracefully rather than raising.

    :param arguments: The tool call arguments dictionary.
    :returns: A canonical string representation suitable for hashing.
    """
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Build the cache key for a tool invocation: `(tool_name, sha256(canonicalized_args_json))`.

    :param tool_name: Name of the tool being invoked.
    :param arguments: The tool call arguments dictionary.
    :returns: A string cache key combining the tool name and a hash of its arguments.
    """
    canonical_args = _canonicalize_args(arguments)
    args_hash = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()
    return f"{tool_name}:{args_hash}"


@dataclass
class ToolCacheStats:
    """
    Tracks cache effectiveness for a single cache instance (typically scoped to one agent run).

    :param hits: Number of cache hits (tool call skipped, cached result returned).
    :param misses: Number of cache misses (tool was actually invoked).
    :param calls_saved: Alias for `hits`, included for readability in run output.
    """

    hits: int = 0
    misses: int = 0

    @property
    def calls_saved(self) -> int:
        """Number of tool invocations avoided due to cache hits."""
        return self.hits

    def to_dict(self) -> dict[str, int]:
        """Serialize stats to a plain dict for inclusion in Agent run output."""
        return {"hits": self.hits, "misses": self.misses, "calls_saved": self.calls_saved}


class ToolCacheBackend(ABC):
    """
    Abstract storage backend for `ToolCache`.

    Implementations only need to handle get/set/clear; TTL expiry and scoping are handled by
    `ToolCache` itself so that every backend gets that behavior for free.
    """

    @abstractmethod
    def get(self, key: str) -> tuple[Any, float] | None:
        """
        Retrieve a cached value and its stored timestamp.

        :param key: The cache key.
        :returns: A tuple of (value, stored_at_unix_timestamp), or None if not present.
        """

    @abstractmethod
    def set(self, key: str, value: Any, stored_at: float) -> None:
        """
        Store a value under the given key with the given timestamp.

        :param key: The cache key.
        :param value: The value to cache (the tool call result).
        :param stored_at: Unix timestamp of when the value was stored.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the backend."""


class InMemoryToolCache(ToolCacheBackend):
    """
    Simple in-process, thread-safe dict-backed cache.

    Not shared across processes and is lost on restart — suitable as the default backend for
    single-process agent runs and as a reference implementation for custom backends.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> tuple[Any, float] | None:
        """See ToolCacheBackend.get()."""
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Any, stored_at: float) -> None:
        """See ToolCacheBackend.set()."""
        with self._lock:
            self._store[key] = (value, stored_at)

    def clear(self) -> None:
        """See ToolCacheBackend.clear()."""
        with self._lock:
            self._store.clear()


class ToolCache:
    """
    Opt-in cache wrapping tool invocations.

    Caching is per-Tool opt-in (`Tool.cacheable`); `ToolCache` itself never decides
    whether a given tool's results may be cached — that decision lives entirely on
    the `Tool` instance so that write-effecting tools cannot accidentally serve a
    cached result for a side-effecting call.

    Usage:
    ```python
    cache = ToolCache(backend=InMemoryToolCache(), ttl_seconds=3600, scope="agent_run")
    agent = Agent(chat_generator=generator, tools=[fetch_url], tool_cache=cache)
    ```

    :param backend: Storage backend. Defaults to a fresh `InMemoryToolCache()`.
    :param ttl_seconds: How long a cached entry remains valid. Defaults to 300 (5 minutes),
        chosen to deduplicate repeated calls within a single agent loop without letting
        staleness compound across longer sessions.
    :param scope: Logical scope label for this cache instance. Purely informational/for the
        caller's own bookkeeping today — `ToolCache` does not enforce cross-scope isolation
        itself; that isolation comes from the caller creating a separate `ToolCache` instance
        per scope (e.g. one per agent run). Accepted values: "agent_run", "session", "global".
    """

    _VALID_SCOPES = ("agent_run", "session", "global")

    def __init__(
        self, backend: ToolCacheBackend | None = None, ttl_seconds: float = 300.0, scope: str = "agent_run"
    ) -> None:
        if scope not in self._VALID_SCOPES:
            msg = f"scope must be one of {self._VALID_SCOPES}, but got {scope!r}"
            raise ValueError(msg)
        if ttl_seconds <= 0:
            msg = f"ttl_seconds must be > 0, but got {ttl_seconds}"
            raise ValueError(msg)

        self.backend = backend if backend is not None else InMemoryToolCache()
        self.ttl_seconds = ttl_seconds
        self.scope = scope
        self.stats = ToolCacheStats()

    def get(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        """
        Look up a cached result for the given tool call.

        :param tool_name: Name of the tool being invoked.
        :param arguments: The tool call arguments dictionary.
        :returns: A tuple `(hit, value)`. If `hit` is False, `value` is always None and the
            caller should invoke the tool normally and call `set()` with the result.
        """
        key = make_cache_key(tool_name, arguments)
        cached = self.backend.get(key)

        if cached is None:
            self.stats.misses += 1
            return False, None

        value, stored_at = cached
        if (time.monotonic() - stored_at) > self.ttl_seconds:
            self.stats.misses += 1
            logger.debug("Tool cache entry for {key} expired (TTL {ttl}s)", key=key, ttl=self.ttl_seconds)
            return False, None

        self.stats.hits += 1
        logger.debug("Tool cache hit for {tool_name}", tool_name=tool_name)
        return True, value

    def set(self, tool_name: str, arguments: dict[str, Any], value: Any) -> None:
        """
        Store a tool call result in the cache.

        :param tool_name: Name of the tool that was invoked.
        :param arguments: The tool call arguments dictionary.
        :param value: The result to cache.
        """
        key = make_cache_key(tool_name, arguments)
        self.backend.set(key, value, time.monotonic())

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self.backend.clear()
        self.stats = ToolCacheStats()
