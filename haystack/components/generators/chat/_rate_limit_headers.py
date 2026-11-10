# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Protocol

import httpx
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from haystack.utils.http_client import init_http_client

_rate_limit_headers: ContextVar[dict[str, str] | None] = ContextVar("_rate_limit_headers", default=None)


class _HttpResponse(Protocol):
    # OpenAI's default client passes an `httpx2` response to the hook, clients built from `http_client_kwargs` an
    # `httpx` one. The hook only reads these two attributes, which both have.
    @property
    def is_success(self) -> bool: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


def _extract_rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """
    Extract the rate limit headers (`x-ratelimit-*`) sent by OpenAI and Azure OpenAI from the HTTP response headers.
    """
    return {name.lower(): value for name, value in headers.items() if name.lower().startswith("x-ratelimit-")}


def _capture_rate_limit_headers(response: _HttpResponse) -> None:
    if response.is_success:
        _rate_limit_headers.set(_extract_rate_limit_headers(response.headers))


async def _async_capture_rate_limit_headers(response: _HttpResponse) -> None:
    _capture_rate_limit_headers(response)


def _reset_rate_limit_headers() -> None:
    """Clear captured headers so concurrent or subsequent runs cannot reuse each other's values."""
    _rate_limit_headers.set(None)


def _captured_rate_limit_headers() -> dict[str, str] | None:
    headers = _rate_limit_headers.get()
    return dict(headers) if headers is not None else None


def _attach_rate_limit_headers(replies: Sequence[Any], *, enabled: bool) -> None:
    """
    Add captured `x-ratelimit-*` headers to each reply's `meta` when capture is enabled.

    An empty dict is added when the successful response had no rate-limit headers. Nothing is added
    when capture is disabled or the hook did not run (for example, the request failed).
    """
    if not enabled:
        return
    headers = _captured_rate_limit_headers()
    if headers is None:
        return
    for reply in replies:
        reply.meta["rate_limit_headers"] = headers


def _http_client_kwargs_with_rate_limit_hook(
    http_client_kwargs: dict[str, Any] | None, *, async_client: bool
) -> dict[str, Any]:
    """
    Return a copy of `http_client_kwargs` that also records rate-limit headers via an httpx response hook.
    """
    kwargs = dict(http_client_kwargs or {})
    existing_hooks = kwargs.get("event_hooks") or {}
    hooks = {key: list(value) for key, value in existing_hooks.items()}
    hook = _async_capture_rate_limit_headers if async_client else _capture_rate_limit_headers
    hooks["response"] = list(hooks.get("response") or []) + [hook]
    kwargs["event_hooks"] = hooks
    return kwargs


def _build_http_client(
    http_client_kwargs: dict[str, Any] | None, *, async_client: bool, include_rate_limit_headers: bool
) -> httpx.Client | httpx.AsyncClient | DefaultHttpxClient | DefaultAsyncHttpxClient | None:
    """
    Build the httpx client used by the OpenAI SDK, attaching a rate-limit hook when requested.

    User-provided `http_client_kwargs` (including existing `event_hooks`) are preserved. When no
    custom kwargs are given, OpenAI's default httpx client is used so timeouts and limits stay the same.
    """
    kwargs = http_client_kwargs
    if include_rate_limit_headers:
        kwargs = _http_client_kwargs_with_rate_limit_hook(http_client_kwargs, async_client=async_client)
        if not http_client_kwargs:
            if async_client:
                return DefaultAsyncHttpxClient(**kwargs)
            return DefaultHttpxClient(**kwargs)
    if async_client:
        return init_http_client(kwargs, async_client=True)
    return init_http_client(kwargs, async_client=False)
