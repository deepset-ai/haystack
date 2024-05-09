# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils.url_validation import is_valid_http_url


def test_url_validation_with_valid_http_url():
    url = "http://example.com"
    assert is_valid_http_url(url)


def test_url_validation_with_valid_https_url():
    url = "https://example.com"
    assert is_valid_http_url(url)


def test_url_validation_with_invalid_scheme():
    url = "ftp://example.com"
    assert not is_valid_http_url(url)


def test_url_validation_with_no_scheme():
    url = "example.com"
    assert not is_valid_http_url(url)


def test_url_validation_with_no_netloc():
    url = "http://"
    assert not is_valid_http_url(url)


def test_url_validation_with_empty_string():
    url = ""
    assert not is_valid_http_url(url)
