# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI

from haystack.utils.http_client import init_http_client


class OpenAIClientMixin:
    """
    Mixin providing OpenAI client lifecycle management.

    Supplies ``_client_kwargs``, ``warm_up``, ``warm_up_async``, ``close``
    and ``close_async`` so that every OpenAI-backed component shares a single
    implementation of these methods.

    Subclasses must set the following attributes **before** any mixin method
    is called (typically at the end of ``__init__``):

    * ``api_key`` – a :class:`~haystack.utils.Secret`
    * ``organization`` – ``str | None``
    * ``api_base_url`` – ``str | None``
    * ``timeout`` – ``float | None``
    * ``max_retries`` – ``int | None``
    * ``http_client_kwargs`` – ``dict[str, Any] | None``
    * ``client`` – initialised to ``None``
    * ``async_client`` – initialised to ``None``
    """

    # Declared here so that mypy knows the mixin expects these on *self*.
    api_key: Any
    organization: str | None
    api_base_url: str | None
    timeout: float | None
    max_retries: int | None
    http_client_kwargs: dict[str, Any] | None
    client: OpenAI | None
    async_client: AsyncOpenAI | None

    def _client_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the OpenAI client constructors."""
        timeout = self.timeout if self.timeout is not None else float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        max_retries = (
            self.max_retries if self.max_retries is not None else int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        )
        return {
            "api_key": self.api_key.resolve_value(),
            "organization": self.organization,
            "base_url": self.api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

    def warm_up(self) -> None:
        """Initializes the synchronous OpenAI client."""
        if hasattr(self, "_warm_up_tools"):
            self._warm_up_tools()
        if self.client is None:
            # openai>=3 annotates http_client as httpx2, but legacy httpx clients are supported at runtime.
            # https://github.com/openai/openai-python/blob/main/httpx2.md
            http_client = init_http_client(self.http_client_kwargs, async_client=False)
            self.client = OpenAI(
                http_client=http_client,  # type: ignore[arg-type]
                **self._client_kwargs(),
            )

    async def warm_up_async(self) -> None:  # noqa: RUF029
        """Initializes the asynchronous OpenAI client on the serving event loop."""
        if hasattr(self, "_warm_up_tools"):
            self._warm_up_tools()
        if self.async_client is None:
            # openai>=3 annotates http_client as httpx2, but legacy httpx clients are supported at runtime.
            # https://github.com/openai/openai-python/blob/main/httpx2.md
            http_client = init_http_client(self.http_client_kwargs, async_client=True)
            self.async_client = AsyncOpenAI(
                http_client=http_client,  # type: ignore[arg-type]
                **self._client_kwargs(),
            )

    def close(self) -> None:
        """Releases the synchronous OpenAI client."""
        if self.client is not None:
            self.client.close()
            self.client = None

    async def close_async(self) -> None:
        """Releases the asynchronous OpenAI client."""
        if self.async_client is not None:
            await self.async_client.close()
            self.async_client = None
