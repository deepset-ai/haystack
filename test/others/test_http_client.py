import time

import pytest
import requests
from requests import RequestException, HTTPError

from haystack.errors import OpenAIUnauthorizedError, OpenAIRateLimitError
from haystack.utils.http_client import HTTPClient


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [413, 429])
def test_retry(httpserver, status_code):
    backoff_factor = 2  # exponential backoff, default value for HTTPClient
    num_retries = 2  # let's make it smaller than default to speed up the test

    httpserver.serve_content(f"{status_code}", status_code)
    c = HTTPClient(url=httpserver.url, method="GET", retries=num_retries)
    start = time.time()
    try:
        c.request(json={"test": "test"})
    except requests.RequestException as e:
        assert isinstance(e, c.default_error_class)
    end = time.time()

    # Check that the retry is working, all retries should take slightly above the total time
    total_time = backoff_factor * (2 ** (num_retries - 1))

    if end - start < total_time:
        pytest.fail("Retries are not working")


@pytest.mark.unit
def test_custom_exception(httpserver):
    httpserver.serve_content("Raise custom OpenAIUnauthorizedError", 401)

    c = HTTPClient(url=httpserver.url, error_codes_map={401: OpenAIUnauthorizedError})
    with pytest.raises(OpenAIUnauthorizedError):
        c.request(json={"test": "test"})


@pytest.mark.unit
def test_default_error_class(httpserver):
    # test that default_error_class works as expected
    # the default HTTPError should be raised

    httpserver.serve_content("NA", 429)

    c = HTTPClient(url=httpserver.url, retries=1)
    with pytest.raises(HTTPError):
        c.request(json={"test": "test"})


@pytest.mark.unit
def test_custom_default_exception(httpserver):
    # test we can raise specific exceptions for specific status codes (after retries)
    httpserver.serve_content("Raise custom OpenAIError", 429)

    c = HTTPClient(url=httpserver.url, retries=2, error_codes_map={429: OpenAIRateLimitError})
    with pytest.raises(OpenAIRateLimitError):
        c.request(json={"test": "test"})


@pytest.mark.unit
def test_default_error_class_none(httpserver):
    # test that default_error_class=None works as expected
    # the default RequestException should be raised

    httpserver.serve_content("NA", 429)

    c = HTTPClient(url=httpserver.url, default_error_class=None, retries=1)
    with pytest.raises(RequestException):
        c.request(json={"test": "test"})
