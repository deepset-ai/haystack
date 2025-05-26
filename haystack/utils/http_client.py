# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Literal, Optional, Union, overload

import httpx


@overload
def init_http_client(http_client_kwargs: Dict[str, Any], async_client: Literal[False]) -> httpx.Client: ...
@overload
def init_http_client(
    http_client_kwargs: Optional[Dict[str, Any]], async_client: Literal[False]
) -> Union[httpx.Client, None]: ...
@overload
def init_http_client(http_client_kwargs: Dict[str, Any], async_client: Literal[True]) -> httpx.AsyncClient: ...
@overload
def init_http_client(
    http_client_kwargs: Optional[Dict[str, Any]], async_client: Literal[True]
) -> Union[httpx.AsyncClient, None]: ...
@overload
def init_http_client(
    http_client_kwargs: Optional[Dict[str, Any]], async_client: bool
) -> Union[httpx.Client, httpx.AsyncClient, None]: ...
def init_http_client(
    http_client_kwargs: Optional[Dict[str, Any]] = None, async_client: bool = False
) -> Union[httpx.Client, httpx.AsyncClient, None]:
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
    if async_client:
        return httpx.AsyncClient(**http_client_kwargs)
    return httpx.Client(**http_client_kwargs)
