# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils.http_client import init_http_client
import httpx


def test_init_http_client():
    # test without any params
    http_client = init_http_client()
    assert http_client is None

    # test client is initialized with http_client_kwargs
    http_client = init_http_client(http_client_kwargs={"base_url": "https://example.com"})
    assert http_client is not None
    assert isinstance(http_client, httpx.Client)
    assert http_client.base_url == "https://example.com"


def test_init_http_client_async():
    # test without any params
    http_async_client = init_http_client(async_client=True)
    assert http_async_client is None

    # test async client is initialized with http_client_kwargs
    http_async_client = init_http_client(http_client_kwargs={"base_url": "https://example.com"}, async_client=True)
    assert http_async_client is not None
    assert isinstance(http_async_client, httpx.AsyncClient)
    assert http_async_client.base_url == "https://example.com"


def test_http_client_kwargs_type_validation():
    # test http_client_kwargs is not a dictionary
    with pytest.raises(TypeError, match="The parameter 'http_client_kwargs' must be a dictionary."):
        init_http_client(http_client_kwargs="invalid")


def test_http_client_kwargs_with_invalid_params():
    # test http_client_kwargs with invalid keys
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        init_http_client(http_client_kwargs={"invalid_key": "invalid"})
