# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.utils.http_client import init_http_client


def test_init_http_client():
    http_client = init_http_client(http_client_kwargs={"base_url": "https://example.com"})
    assert http_client is not None
    assert http_client.base_url == "https://example.com"


def test_http_client_kwargs_type_validation():
    with pytest.raises(TypeError, match="The parameter 'http_client_kwargs' must be a dictionary."):
        init_http_client(http_client_kwargs="invalid")


def test_http_client_kwargs_with_invalid_params():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        init_http_client(http_client_kwargs={"invalid_key": "invalid"})
