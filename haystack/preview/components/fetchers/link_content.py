import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests import Response
from requests.exceptions import HTTPError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from haystack import __version__
from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.dataclasses import ByteStream

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENT = f"haystack/LinkContentFetcher/{__version__}"

REQUEST_HEADERS = {
    "accept": "*/*",
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
    "referer": "https://www.google.com/",
}


def text_content_handler(response: Response) -> ByteStream:
    """
    :param response: Response object from the request.
    :return: The extracted text.
    """
    return ByteStream.from_string(response.text)


def binary_content_handler(response: Response) -> ByteStream:
    """
    :param response: Response object from the request.
    :return: The extracted binary file-like object.
    """
    return ByteStream(data=response.content)


@component
class LinkContentFetcher:
    """
    LinkContentFetcher is a component for fetching and extracting content from URLs. It supports handling various
    content types, retries on failures, and automatic user-agent rotation for failed web requests.
    """

    def __init__(
        self,
        raise_on_failure: bool = True,
        user_agents: Optional[List[str]] = None,
        retry_attempts: int = 2,
        timeout: int = 3,
    ):
        """
        Creates a LinkContentFetcher instance.

        :param raise_on_failure: A boolean indicating whether to raise an exception when a failure occurs
            during content extraction. If False, the error is simply logged and the program continues.
            Defaults to False.
        :param user_agents: A list of user agents to use when fetching content. Defaults to None, in which case a
            default user agent is used.
        :param retry_attempts: The number of times to retry fetching content. Defaults to 2.
        :param timeout: The timeout in seconds for the request. Defaults to 3.
        """
        self.raise_on_failure = raise_on_failure
        self.user_agents = user_agents or [DEFAULT_USER_AGENT]
        self.current_user_agent_idx: int = 0
        self.retry_attempts = retry_attempts
        self.timeout = timeout

        # register default content handlers that extract data from the response
        self.handlers: Dict[str, Callable[[Response], ByteStream]] = defaultdict(lambda: text_content_handler)
        self.handlers["text/html"] = text_content_handler
        self.handlers["text/plain"] = text_content_handler
        self.handlers["application/pdf"] = binary_content_handler
        self.handlers["application/octet-stream"] = binary_content_handler

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=(retry_if_exception_type((HTTPError, requests.RequestException))),
            # This method is invoked only after failed requests (exception raised)
            after=self._switch_user_agent,
        )
        def get_response(url):
            # we need to copy because we modify the headers
            headers = REQUEST_HEADERS.copy()
            headers["User-Agent"] = self.user_agents[self.current_user_agent_idx]
            response = requests.get(url, headers=headers, timeout=timeout or 3)
            response.raise_for_status()
            return response

        self._get_response: Callable = get_response

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            raise_on_failure=self.raise_on_failure,
            user_agents=self.user_agents,
            retry_attempts=self.retry_attempts,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkContentFetcher":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(streams=List[ByteStream])
    def run(self, urls: List[str]):
        """
        Fetches content from a list of URLs and returns a list of extracted content streams.
        Each content stream is a ByteStream object containing the extracted content as binary data.
        The content type of each stream is stored in the metadata of the ByteStream object under
        the key "content_type".

        :param urls: A list of URLs to fetch content from.
        :return: A lists of ByteStream objects representing the extracted content.
        """
        streams: List[ByteStream] = []
        if not urls:
            return {"streams": streams}

        # don't use multithreading if there's only one URL
        if len(urls) == 1:
            content_type, stream = self.fetch(urls[0])
            if content_type and stream:
                stream.metadata["content_type"] = content_type
                streams.append(stream)
        else:
            with ThreadPoolExecutor() as executor:
                results = executor.map(self.fetch, urls)

            for content_type, stream in results:
                if content_type and stream:
                    stream.metadata["content_type"] = content_type
                    streams.append(stream)

        return {"streams": streams}

    def fetch(self, url: str) -> Tuple[str, ByteStream]:
        """
        Fetches content from a URL and returns it as a ByteStream.

        :param url: The URL to fetch content from.
        :return: A tuple containing the content type and the corresponding ByteStream.
             The content type is a string indicating the type of content fetched (e.g., "text/html", "application/pdf").
             The ByteStream object contains the fetched content as binary data.

        :raises: If an error occurs during content retrieval and `raise_on_failure` is set to True, this method will
        raise an exception. Otherwise, errors are logged, and an empty ByteStream is returned.

        """
        content_type: str = "text/html"
        stream: ByteStream = ByteStream(data=b"")
        try:
            response = self._get_response(url)
            content_type = self._get_content_type(response)
            handler: Callable = self.handlers[content_type]
            stream = handler(response)
        except Exception as e:
            if self.raise_on_failure:
                raise e
            # less verbose log as this is expected to happen often (requests failing, blocked, etc.)
            logger.debug("Couldn't retrieve content from %s due to %s", url, str(e))

        finally:
            self.current_user_agent_idx = 0

        return content_type, stream

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
        Used by tenacity to retry the requests with a different user agent.

        :param retry_state: The retry state (unused, required by tenacity).
        """
        self.current_user_agent_idx = (self.current_user_agent_idx + 1) % len(self.user_agents)
        logger.debug("Switched user agent to %s", self.user_agents[self.current_user_agent_idx])
