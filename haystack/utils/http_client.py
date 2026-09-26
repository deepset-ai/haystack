# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, overload

import httpx2


@overload
def init_http_client(
    http_client_kwargs: dict[str, Any] | None = ..., async_client: Literal[False] = ...
) -> httpx2.Client | None: ...
@overload
def init_http_client(
    http_client_kwargs: dict[str, Any] | None = ..., async_client: Literal[True] = ...
) -> httpx2.AsyncClient | None: ...
def init_http_client(
    http_client_kwargs: dict[str, Any] | None = None, async_client: bool = False
) -> httpx2.Client | httpx2.AsyncClient | None:
    """
    Initialize an httpx2 client based on the http_client_kwargs.

    :param http_client_kwargs:
        The kwargs to pass to the httpx2 client.
    :param async_client:
        Whether to initialize an async client.

    :returns:
        A httpx2 client or an async httpx2 client.
    """
    if not http_client_kwargs:
        return None
    if not isinstance(http_client_kwargs, dict):
        raise TypeError("The parameter 'http_client_kwargs' must be a dictionary.")

    # Create a copy to avoid modifying the original dict
    processed_kwargs = http_client_kwargs.copy()

    # Handle limits parameter - convert dict to httpx2.Limits object if needed
    if "limits" in processed_kwargs and isinstance(processed_kwargs["limits"], dict):
        limits_dict = processed_kwargs["limits"]
        processed_kwargs["limits"] = httpx2.Limits(**limits_dict)

    if async_client:
        return httpx2.AsyncClient(**processed_kwargs)
    return httpx2.Client(**processed_kwargs)
