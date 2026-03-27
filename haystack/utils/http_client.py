# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, overload

import httpx


@overload
def init_http_client(http_client_kwargs: dict[str, Any], async_client: Literal[False]) -> httpx.Client: ...
@overload
def init_http_client(
    http_client_kwargs: dict[str, Any] | None, async_client: Literal[False]
) -> httpx.Client | None: ...
@overload
def init_http_client(http_client_kwargs: dict[str, Any], async_client: Literal[True]) -> httpx.AsyncClient: ...
@overload
def init_http_client(
    http_client_kwargs: dict[str, Any] | None, async_client: Literal[True]
) -> httpx.AsyncClient | None: ...
@overload
def init_http_client(
    http_client_kwargs: dict[str, Any] | None, async_client: bool
) -> httpx.Client | httpx.AsyncClient | None: ...
def init_http_client(
    http_client_kwargs: dict[str, Any] | None = None, async_client: bool = False
) -> httpx.Client | httpx.AsyncClient | None:
    """
    Initialize an httpx client based on the http_client_kwargs.

    :param http_client_kwargs:
        The kwargs to pass to the httpx client.
    :param async_client:
        Whether to initialize an async client.

    :returns:
        A httpx client or an async httpx client.
    """
    if not http_client_kwargs:
        return None
    if not isinstance(http_client_kwargs, dict):
        raise TypeError("The parameter 'http_client_kwargs' must be a dictionary.")

    # Create a copy to avoid modifying the original dict
    processed_kwargs = http_client_kwargs.copy()

    # Handle limits parameter - convert dict to httpx.Limits object if needed
    if "limits" in processed_kwargs and isinstance(processed_kwargs["limits"], dict):
        limits_dict = processed_kwargs["limits"]
        processed_kwargs["limits"] = httpx.Limits(**limits_dict)

    if async_client:
        return httpx.AsyncClient(**processed_kwargs)
    return httpx.Client(**processed_kwargs)
