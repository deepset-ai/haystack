import os
from typing import Optional, Callable, Any, Set, Dict, Type

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util.retry import Retry

from haystack.environment import HAYSTACK_REMOTE_API_TIMEOUT_SEC, HAYSTACK_REMOTE_API_MAX_RETRIES

HTTP_TIMEOUT = int(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
HTTP_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


class HTTPClient:
    """
    HTTPClient provides a simplified interface for sending HTTP requests to a specified URL with the ability to set
    custom request headers, timeouts, retries for specific status codes and error handling.

    Example Usage:
    --------------

    # Creating an instance of HTTPClient
    from haystack.utils.http_client import HTTPClient
    client = HTTPClient(url="http://localhost:8080/test", method="GET", headers={"Authorization": "Bearer my_token"},
                        retries=3, timeout=10)

    # Sending an HTTP request with retries
    response = client.request(json={"param1": "value1", "param2": "value2"})
    print(response.status_code)

    # Handling a custom error with a custom exception
     from haystack.errors import OpenAIUnauthorizedError
     client = HTTPClient(url="http://localhost:8080/test", method="GET", error_codes_map={401: OpenAIUnauthorizedError})
     try:
         client.request(json={"test": "test"})
     except OpenAIUnauthorizedError as e:
         print(e)

    # Handling of custom retry status codes and using a custom error class for 429 status code
    from haystack.utils.http_client import HTTPClient
    from haystack.errors import OpenAIRateLimitError
    retry_status_list = {429, 503}

    # Create an instance of the HTTPClient with custom retry status codes
    client = HTTPClient(
        url="https://api.openai.com/v1/completions",
        method="GET",
        retries=3,
        retry_status_list=retry_status_list,
        error_codes_map={429: OpenAIRateLimitError}
    )

    # Make a request using the HTTPClient instance
    try:
        response = client.request(json={"test": "test"})
        print(response.status_code)
    except Exception as e:
        print(f"Error: {e}")


    # Handling a custom error with the default exception
     from requests import HTTPError
     client = HTTPClient(url="http://localhost:8080/test", method="GET")
     try:
         client.request(json={"test": "test"})
     except HTTPError as e:
         print(e)

    # Handling a timeout error
     client = HTTPClient(url="http://localhost:8080/test", method="GET", timeout=1)
     try:
         client.request(json={"test": "test"})
     except TimeoutError as e:
         print(e)

    """

    def __init__(
        self,
        url: str,
        method: Optional[str] = "GET",
        headers: Optional[dict] = None,
        timeout: Optional[int] = HTTP_TIMEOUT,
        retries: Optional[int] = HTTP_MAX_RETRIES,
        backoff_factor: Optional[float] = 2.0,
        retry_status_list: Optional[Set[int]] = None,
        response_handler: Optional[Callable[[requests.Response], Any]] = None,
        error_codes_map: Optional[Dict[int, Type]] = None,
        default_error_class: Optional[Type] = HTTPError,
        pool_connections: Optional[int] = 2,
        pool_maxsize: Optional[int] = 2,
    ):
        """
        Create a new HTTPClient with a requests Session preconfigured with retry and timeout functionality.

        :param method: HTTP method to use for the request.
        :param url: URL to send the request to.
        :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
        :param timeout: The timeout length of the request. The default is 30s.
        :param retries: The number of retries to attempt. The default is 5.
        :param backoff_factor: The backoff factor to use when retrying. The default is 2 (a.k.a. exponential backoff).
        :param retry_status_list: The list of status codes to retry on. The default is [500, 502, 503, 504, 413, 429].
        :param response_handler: A function to handle the response. The default is to return the response as JSON.
        :param error_codes_map: A dictionary of error codes to exception classes. The default is to raise an HTTPError.
        :param default_error_class: The default exception class to raise. The default is HTTPError.
        :param pool_connections: The number of connections to keep in the pool. The default is 2.
        :param pool_maxsize: The maximum number of connections to save in the pool. The default is 2.
        """
        self.session = requests.Session()
        self.method = method
        self.url = url
        self.retry_status_list = retry_status_list or {500, 502, 503, 504, 413, 429}
        retry_strategy = Retry(
            total=retries, backoff_factor=backoff_factor, status_forcelist=self.retry_status_list, raise_on_status=False  # type: ignore
        )
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=retry_strategy)  # type: ignore
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.timeout = timeout
        self.error_codes_map = error_codes_map or {}
        self.default_error_class = default_error_class
        self.response_handler = response_handler if response_handler else lambda response: response.json()
        self.headers = headers or {}

    def request(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        stream: Optional[bool] = False,
        response_handler: Optional[Callable[[requests.Response], Any]] = None,
    ) -> Any:
        """
        Send a request using the configured session with an option to override the method, url, headers, and response handler.

        :param method: HTTP method to use for the request.
        :param url: URL to send the request to.
        :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
        :param params: Dictionary or bytes to be sent in the query string for the :class:`Request`.
        :param data: Dictionary, bytes, or file-like object to send in the body of the :class:`Request`.
        :param json: JSON serializable Python object to send in the body of the :class:`Request`.
        :param stream: Whether to immediately download the response content. The default is False.
        :param response_handler: A function to handle the response. The default is to return the response as JSON.
        :return: The response from the request.
        """
        try:
            res = self.session.request(
                method or self.method,  # type: ignore
                url or self.url,
                params=params,
                data=data,
                json=json,
                headers=headers or self.headers,
                timeout=self.timeout,
                stream=stream,
            )
            handler = response_handler or self.response_handler
            if res.status_code != 200:
                exc_class = self.error_codes_map.get(res.status_code, self.default_error_class)
                if exc_class:
                    raise exc_class(f"Request failed with status code {res.status_code}")

                res.raise_for_status()

            return handler(res)
        except requests.exceptions.RequestException as e:
            raise self.default_error_class(f"Request failed {e}") if self.default_error_class else e
