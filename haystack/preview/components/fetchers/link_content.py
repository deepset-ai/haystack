import io
import logging
from collections import defaultdict
from datetime import datetime
from http import HTTPStatus
from typing import Optional, Dict, List, Callable, Any, IO
from urllib.parse import urlparse

import requests
from requests import Response
from requests.exceptions import InvalidURL, HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState
from haystack.preview import component, default_from_dict, default_to_dict

from haystack import __version__
from haystack.preview import Document

logger = logging.getLogger(__name__)


def text_content_handler(response: Response) -> Optional[str]:
    """
    :param response: Response object from the request.
    :return: The extracted text.
    """
    return response.text


def binary_content_handler(response: Response) -> IO[bytes]:
    """
    :param response: Response object from the request.
    :return: The extracted binary file-like object.
    """
    return io.BytesIO(response.content)


@component
class LinkContentFetcher:
    """
    LinkContentFetcher fetches content from a URL link and converts it to a Document object.
    """

    _USER_AGENT = f"haystack/LinkContentFetcher/{__version__}"

    _REQUEST_HEADERS = {
        "accept": "*/*",
        "User-Agent": _USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
        "referer": "https://www.google.com/",
    }

    def __init__(
        self,
        raise_on_failure: Optional[bool] = True,
        user_agents: Optional[List[str]] = None,
        retry_attempts: Optional[int] = None,
    ):
        """

        Creates a LinkContentFetcher instance.
        :param raise_on_failure: A boolean indicating whether to raise an exception when a failure occurs
                         during content extraction. If False, the error is simply logged and the program continues.
                         Defaults to False.
        :param user_agents: A list of user agents to use when fetching content. Defaults to None, in which case a
        default user agent is used.
        :param retry_attempts: The number of times to retry fetching content. Defaults to 2.
        """
        super().__init__()
        self.raise_on_failure = raise_on_failure
        self.user_agents = user_agents or [LinkContentFetcher._USER_AGENT]
        self.current_user_agent_idx: int = 0
        self.retry_attempts = retry_attempts or 2
        self.handlers: Dict[str, Callable] = defaultdict(lambda: text_content_handler)

        # register default content handlers that extract data from the response
        self.handlers["text/html"] = text_content_handler
        self.handlers["text/plain"] = text_content_handler
        self.handlers["application/pdf"] = binary_content_handler

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            raise_on_failure=self.raise_on_failure,
            user_agents=self.user_agents,
            retry_attempts=self.retry_attempts,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkContentFetcher":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def run(self, url: str, timeout: Optional[int] = 3, doc_kwargs: Optional[dict] = None) -> Document:
        """
        Fetches content from a URL and converts it to a Document objects. If no content is extracted,
        an empty Document object is returned (if raise_on_failure is False).

        :param url: URL to fetch content from.
        :param timeout: Timeout in seconds for the request.
        :param doc_kwargs: Optional kwargs to pass to the Document constructor.
        :return: List of Document objects or an empty list if no content is extracted.
        """
        if not self._is_valid_url(url):
            raise InvalidURL("Invalid or missing URL: {}".format(url))

        doc_kwargs = doc_kwargs or {}
        extracted_doc: Dict[str, Any] = {"metadata": {"url": url, "timestamp": int(datetime.utcnow().timestamp())}}
        extracted_doc.update(doc_kwargs)
        response = self._get_response(url, timeout=timeout or 3)
        has_content = response.status_code == HTTPStatus.OK and (response.text or response.content)
        if has_content:
            content_type = self._get_content_type(response)
            handler: Callable = self.handlers[content_type]
            extracted_content = handler(response)
            extracted_doc["content"] = extracted_content
            # TODO assign content_type to created document
            return Document(**extracted_doc)
        else:
            if self.raise_on_failure:
                raise Exception(f"Couldn't retrieve content from {url}")
            return Document(content="")

    def _get_response(self, url: str, timeout: Optional[int] = None) -> requests.Response:
        """
        Fetches content from a URL. Returns a response object.
        :param url: The URL to fetch content from.
        :param timeout: The timeout in seconds.
        :return: A response object.
        """

        @retry(
            # we want to reraise the exception if we fail after the last self.retry_attempts
            # then we can catch it in the outer try/except block, see below
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=(retry_if_exception_type((HTTPError, requests.RequestException))),
            # This method is invoked only after failed requests (exception raised)
            after=self._switch_user_agent,
        )
        def _request():
            # we need a request copy because we modify the headers
            headers = self._REQUEST_HEADERS.copy()
            headers["User-Agent"] = self.user_agents[self.current_user_agent_idx]
            r = requests.get(url, headers=headers, timeout=timeout or 3)
            r.raise_for_status()
            return r

        try:
            response = _request()
        except Exception as e:
            # catch all exceptions including HTTPError and RequestException
            if self.raise_on_failure:
                raise e
            # if we don't raise on failure, log it, and return a response object
            logger.warning("Couldn't retrieve content from %s", url)
            response = requests.Response()
        finally:
            self.current_user_agent_idx = 0
        return response

    def _get_content_type(self, response: Response):
        """
        Get the content type of the response.
        :param response: The response object.
        :return: The content type of the response.
        """
        content_type = response.headers.get("Content-Type", "")
        return content_type.split(";")[0]

    def _switch_user_agent(self, retry_state: RetryCallState) -> None:
        """
        Switches the User-Agent for this LinkContentRetriever to the next one in the list of user agents.
        :param retry_state: The retry state (unused, required by tenacity).
        """
        self.current_user_agent_idx = (self.current_user_agent_idx + 1) % len(self.user_agents)

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks if a URL is valid.

        :param url: The URL to check.
        :return: True if the URL is valid, False otherwise.
        """

        result = urlparse(url)
        # schema is http or https and netloc is not empty
        return all([result.scheme in ["http", "https"], result.netloc])
