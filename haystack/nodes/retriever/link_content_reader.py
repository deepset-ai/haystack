import logging
import mimetypes
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import requests
from boilerpy3 import extractors
from haystack import Document, __version__
from haystack.nodes import PreProcessor

logger = logging.getLogger(__name__)


class LinkContentReader:
    """
    LinkContentReader fetches content from a URL and converts it into a list of Document objects.

    LinkContentReader supports content types:
    - HTML

    """

    def __init__(self, url: Optional[str] = None, pre_processor: Optional[PreProcessor] = None):
        """
        Creates a LinkContentReader instance
        :param url: URL to fetch content from
        :param pre_processor: PreProcessor to apply to the extracted text
        """
        self.url = url
        self.pre_processor = pre_processor
        self.handlers = {"default": self.fetch_default}

    def __call__(self, **kwargs) -> List[Document]:
        """
        Fetches content from a URL and converts it into a list of Document objects.
        """
        url = self.url or kwargs.get("url")
        if not url:
            raise ValueError("LinkContentReader requires a url parameter to be set either during init or at runtime.")

        timeout = kwargs.get("timeout") or 3
        doc_kwargs = kwargs.get("doc_kwargs", {})

        extracted_doc = {}
        handler = self.get_handler(url, timeout)
        if handler in self.handlers:
            extracted_doc = self.handlers[handler](url, timeout)

        if not extracted_doc:
            logger.debug("Could not extract text from URL %s.", url)

        # we proceed because we might have some content provided via doc_kwargs (e.g. snippets from a search engine)
        extracted_doc = {**doc_kwargs, **extracted_doc}
        document = Document.from_dict(extracted_doc, field_map={"text": "content"})
        document.meta["timestamp"] = int(datetime.utcnow().timestamp())
        if self.pre_processor:
            return self.pre_processor.process(documents=[document])
        return [document]

    def fetch_default(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Fetches content from a URL and returns it as a Document object. The default handler is used when the URL content
        type is HTML or could not be determined.
        """
        extracted_doc = {"url": url}
        extractor = extractors.ArticleExtractor(raise_on_failure=False)

        if not self.is_valid_url(url):
            logger.debug("Invalid URL: %s", url)
            return extracted_doc

        try:
            response = requests.get(url, headers=self.request_headers(), timeout=timeout)
        except requests.RequestException as e:
            # these http errors during content fetching are not so rare, we can continue with empty content
            logger.debug("Error retrieving URL %s: %s", url, e)
            return extracted_doc

        if response.status_code == 200 and len(response.text) > 0:
            try:
                extracted_content = extractor.get_content(response.text)
                if extracted_content:
                    extracted_doc = {**extracted_doc, "text": extracted_content}
            except Exception as e:
                # these parsing errors occur as well, we can continue with empty content
                logger.debug("Error extracting content from URL %s: %s", url, e)

        return extracted_doc

    def get_handler(self, url: str, timeout: int = 10) -> str:
        """
        Determines the appropriate handler to use for fetching content from the provided URL.

        This determination is made based on the Content-Type header returned by the server. If the Content-Type header
        can be mapped to a file extension that isn't ".htm" or ".html", this file extension is used as the handler.
        If no Content-Type header was provided, or if the Content-Type couldn't be mapped to a valid handler,
        the function returns "default".

        Note: A HEAD request is used to fetch the Content-Type header. In case of any exception during this request,
        it's caught and logged, and the function continues by returning "default".

        :param url: The URL from which to fetch content.
        :param timeout: The timeout duration for the HEAD request in seconds.
        :return: The determined handler as a string.
        """
        handler = "default"

        try:
            # Send a HEAD request to the URL and extract the Content-Type header
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            content_type = response.headers.get("content-type")

            # If the server provided a Content-Type header, guess the file extension
            if content_type:
                ext = mimetypes.guess_extension(content_type.split(";")[0])

                # If the MIME type could be mapped to an extension and the extension is not .htm or .html,
                # return the extension
                if ext and ext not in {".htm", ".html"}:
                    handler = ext[1:]

        except requests.exceptions.RequestException:
            logger.debug("Failed to fetch content type from URL %s.", url)

        return handler

    def is_valid_url(self, url: str) -> bool:
        """
        Checks if a URL is valid.

        :param url: The URL to check.
        :return: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def request_headers(self):
        return {
            "accept": "*/*",
            "User-Agent": f"haystack/LinkContentReader/{__version__}",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
            "referer": "https://www.google.com/",
        }
