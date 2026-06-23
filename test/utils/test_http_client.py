# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from haystack.utils.http_client import init_http_client


def test_init_http_client():
    # test without any params (relies on the implementation's default args, which the overloads do not expose)
    http_client = init_http_client()  # type: ignore[call-overload]  # valid no-arg call; overloads omit defaults
    assert http_client is None

    # test client is initialized with http_client_kwargs (async_client correctly defaults to False at runtime,
    # but every overload requires it to be passed explicitly)
    client_kwargs = {"base_url": "https://example.com"}
    http_client = init_http_client(http_client_kwargs=client_kwargs)  # type: ignore[call-overload]
    assert http_client is not None
    assert isinstance(http_client, httpx.Client)
    assert http_client.base_url == "https://example.com"


def test_init_http_client_async():
    # test without any params (http_client_kwargs correctly defaults to None at runtime,
    # but every overload requires it to be passed explicitly)
    http_async_client = init_http_client(async_client=True)  # type: ignore[call-overload]
    assert http_async_client is None

    # test async client is initialized with http_client_kwargs
    http_async_client = init_http_client(http_client_kwargs={"base_url": "https://example.com"}, async_client=True)
    assert http_async_client is not None
    assert isinstance(http_async_client, httpx.AsyncClient)
    assert http_async_client.base_url == "https://example.com"


def test_http_client_kwargs_type_validation():
    # test http_client_kwargs is not a dictionary
    with pytest.raises(TypeError, match="The parameter 'http_client_kwargs' must be a dictionary."):
        init_http_client(http_client_kwargs="invalid")  # type: ignore[call-overload]  # intentionally invalid type


def test_http_client_kwargs_with_invalid_params():
    # test http_client_kwargs with invalid keys (async_client defaults to False at runtime, but overloads require it)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        init_http_client(http_client_kwargs={"invalid_key": "invalid"})  # type: ignore[call-overload]


def test_init_http_client_with_dict_limits():
    """Test that dict limits are converted to httpx.Limits objects without AttributeError."""
    http_client_kwargs = {"limits": {"max_connections": 100, "max_keepalive_connections": 20}}

    # This should not raise AttributeError: 'dict' object has no attribute 'max_connections'
    client = init_http_client(http_client_kwargs=http_client_kwargs, async_client=False)
    assert client is not None
    assert isinstance(client, httpx.Client)

    client.close()
