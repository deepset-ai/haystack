# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, List, Optional

import httpx
import requests
from tenacity import after_log, before_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__file__)


def request_with_retry(
    attempts: int = 3, status_codes_to_retry: Optional[List[int]] = None, **kwargs: Any
) -> requests.Response:
    """
    Executes an HTTP request with a configurable exponential backoff retry on failures.

    Usage example:
    ```python
    from haystack.utils import request_with_retry

    # Sending an HTTP request with default retry configs
    res = request_with_retry(method="GET", url="https://example.com")

    # Sending an HTTP request with custom number of attempts
    res = request_with_retry(method="GET", url="https://example.com", attempts=10)

    # Sending an HTTP request with custom HTTP codes to retry
    res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[408, 503])

    # Sending an HTTP request with custom timeout in seconds
    res = request_with_retry(method="GET", url="https://example.com", timeout=5)

    # Sending an HTTP request with custom authorization handling
    class CustomAuth(requests.auth.AuthBase):
        def __call__(self, r):
            r.headers["authorization"] = "Basic <my_token_here>"
            return r

    res = request_with_retry(method="GET", url="https://example.com", auth=CustomAuth())

    # All of the above combined
    res = request_with_retry(
        method="GET",
        url="https://example.com",
        auth=CustomAuth(),
        attempts=10,
        status_codes_to_retry=[408, 503],
        timeout=5
    )

    # Sending a POST request
    res = request_with_retry(method="POST", url="https://example.com", data={"key": "value"}, attempts=10)

    # Retry all 5xx status codes
    res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=list(range(500, 600)))
    ```

    :param attempts:
        Maximum number of attempts to retry the request.
    :param status_codes_to_retry:
        List of HTTP status codes that will trigger a retry.
        When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
    :param kwargs:
        Optional arguments that `request` accepts.
    :returns:
        The `Response` object.
    """

    if status_codes_to_retry is None:
        status_codes_to_retry = [408, 418, 429, 503]

    @retry(
        reraise=True,
        wait=wait_exponential(),
        retry=retry_if_exception_type((requests.HTTPError, TimeoutError)),
        stop=stop_after_attempt(attempts),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    def run():
        timeout = kwargs.pop("timeout", 10)
        res = requests.request(**kwargs, timeout=timeout)

        if res.status_code in status_codes_to_retry:
            # We raise only for the status codes that must trigger a retry
            res.raise_for_status()

        return res

    res = run()
    # We raise here too in case the request failed with a status code that
    # won't trigger a retry, this way the call will still cause an explicit exception
    res.raise_for_status()
    return res


async def async_request_with_retry(
    attempts: int = 3, status_codes_to_retry: Optional[List[int]] = None, **kwargs: Any
) -> httpx.Response:
    """
    Executes an asynchronous HTTP request with a configurable exponential backoff retry on failures.

    Usage example:
    ```python
    import asyncio
    from haystack.utils import async_request_with_retry

    # Sending an async HTTP request with default retry configs
    async def example():
        res = await async_request_with_retry(method="GET", url="https://example.com")
        return res

    # Sending an async HTTP request with custom number of attempts
    async def example_with_attempts():
        res = await async_request_with_retry(method="GET", url="https://example.com", attempts=10)
        return res

    # Sending an async HTTP request with custom HTTP codes to retry
    async def example_with_status_codes():
        res = await async_request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[408, 503])
        return res

    # Sending an async HTTP request with custom timeout in seconds
    async def example_with_timeout():
        res = await async_request_with_retry(method="GET", url="https://example.com", timeout=5)
        return res

    # Sending an async HTTP request with custom headers
    async def example_with_headers():
        headers = {"Authorization": "Bearer <my_token_here>"}
        res = await async_request_with_retry(method="GET", url="https://example.com", headers=headers)
        return res

    # All of the above combined
    async def example_combined():
        headers = {"Authorization": "Bearer <my_token_here>"}
        res = await async_request_with_retry(
            method="GET",
            url="https://example.com",
            headers=headers,
            attempts=10,
            status_codes_to_retry=[408, 503],
            timeout=5
        )
        return res

    # Sending an async POST request
    async def example_post():
        res = await async_request_with_retry(
            method="POST",
            url="https://example.com",
            json={"key": "value"},
            attempts=10
        )
        return res

    # Retry all 5xx status codes
    async def example_5xx():
        res = await async_request_with_retry(
            method="GET",
            url="https://example.com",
            status_codes_to_retry=list(range(500, 600))
        )
        return res
    ```

    :param attempts:
        Maximum number of attempts to retry the request.
    :param status_codes_to_retry:
        List of HTTP status codes that will trigger a retry.
        When param is `None`, HTTP 408, 418, 429 and 503 will be retried.
    :param kwargs:
        Optional arguments that `httpx.AsyncClient.request` accepts.
    :returns:
        The `httpx.Response` object.
    """

    if status_codes_to_retry is None:
        status_codes_to_retry = [408, 418, 429, 503]

    @retry(
        reraise=True,
        wait=wait_exponential(),
        retry=retry_if_exception_type((httpx.HTTPError, TimeoutError)),
        stop=stop_after_attempt(attempts),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    async def run():
        timeout = kwargs.pop("timeout", 10)
        async with httpx.AsyncClient() as client:
            res = await client.request(**kwargs, timeout=timeout)

            if res.status_code in status_codes_to_retry:
                # We raise only for the status codes that must trigger a retry
                res.raise_for_status()

            return res

    res = await run()
    # We raise here too in case the request failed with a status code that
    # won't trigger a retry, this way the call will still cause an explicit exception
    res.raise_for_status()
    return res
