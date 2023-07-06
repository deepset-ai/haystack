import logging
from datetime import datetime
from http import HTTPStatus
from typing import Optional, Dict, List, Union, Callable
from urllib.parse import urlparse

import requests
from boilerpy3 import extractors
from requests import Response
from requests.exceptions import InvalidURL

from haystack import Document, __version__
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import FilterType

logger = logging.getLogger(__name__)


def html_content_handler(response: Response) -> Optional[str]:
    """
    Extracts content from the response text using the boilerpy3 extractor.
    """
    extractor = extractors.ArticleExtractor(raise_on_failure=False)
    try:
        extracted_content = extractor.get_content(response.text)
        return extracted_content
    except Exception:
        return None


def pdf_content_handler(response: Response) -> Optional[str]:
    # TODO: implement this
    return None


class LinkContentRetriever(BaseRetriever):
    """
    LinkContentRetriever fetches content from a URL and converts it into a list of Document objects.

    LinkContentRetriever supports the following content types:
    - HTML

    """

    def __init__(self, pre_processor: Optional[PreProcessor] = None):
        """
        Creates a LinkContentRetriever instance.
        :param pre_processor: PreProcessor to apply to the extracted text
        """
        super().__init__()
        self.pre_processor = pre_processor
        self.content_handlers: Dict[str, Callable] = {"html": html_content_handler, "pdf": pdf_content_handler}

    def __call__(self, url: str, timeout: int = 3, doc_kwargs: Optional[dict] = None) -> List[Document]:
        """
        Fetches content from a URL and converts it into a list of Document objects.
        :param url: URL to fetch content from.
        :param timeout: Timeout in seconds for the request.
        :param doc_kwargs: Optional kwargs to pass to the Document constructor.
        :return: List of Document objects.
        """
        if not url or not self._is_valid_url(url):
            raise InvalidURL("Invalid or missing URL: {}".format(url))

        doc_kwargs = doc_kwargs or {}
        extracted_doc = {"url": url}

        response = self._get_response(url, timeout)
        if not response:
            return []

        # will handle non-HTML content types soon, add content type resolution here
        handler = "html"
        if handler in self.content_handlers:
            extracted_content = self.content_handlers[handler](response)
            if extracted_content:
                extracted_doc.update({"text": extracted_content})
            else:
                logger.debug("Couldn't extract content from URL %s, using content handler %s.", url, handler)
                return []

        extracted_doc.update(doc_kwargs)
        document = Document.from_dict(extracted_doc, field_map={"text": "content"})
        document.meta["timestamp"] = int(datetime.utcnow().timestamp())

        if self.pre_processor:
            return self.pre_processor.process(documents=[document])

        return [document]

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Fetches content from a URL specified by query parameter and converts it into a list of Document objects.

        param query: The query - a URL to fetch content from.
        param filters: Not used.
        param top_k: Return only the top_k results. If None, the top_k value passed to the constructor is used.
        param index: Not used.
        param headers: Not used.
        param scale_score: Not used.
        param document_store: Not used.

        return: List of Document objects.
        """
        if not query:
            raise ValueError("LinkContentRetriever run requires the `query` parameter")
        return self(url=query)

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
        top_k: Optional[int] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        scale_score: Optional[bool] = None,
        document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Takes a list of queries, where each query is expected to be a URL. For each query, the method
        fetches content from the specified URL and transforms it into a list of Document objects. The output is a list
        of these document lists, where each individual list of Document objects corresponds to the content retrieved
        from a specific query URL.

        param queries: List of queries - URLs to fetch content from.
        param filters: Not used.
        param top_k: Return only the top_k results. If None, the top_k value passed to the constructor is used.
        param index: Not used.
        param headers: Not used.
        param batch_size: Not used.
        param scale_score: Not used.
        param document_store: Not used.

        return: List of lists of Document objects.
        """
        results = []
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            raise ValueError(
                "LinkContentRetriever run_batch requires the `queries` parameter to be Union[str, List[str]]"
            )
        for query in queries:
            results.append(self(url=query))
        return results

    def _get_response(self, url: str, timeout: int) -> Optional[requests.Response]:
        """
        Fetches content from a URL. Returns a response object.
        :param url: The URL to fetch content from.
        :param timeout: The timeout in seconds.
        :return: A response object.
        """
        try:
            response = requests.get(url, headers=self._request_headers(), timeout=timeout)
            if response.status_code != HTTPStatus.OK or len(response.text) == 0:
                logger.debug("Error retrieving URL %s: Status Code - %s", url, response.status_code)
                return None
            return response
        except requests.RequestException as e:
            logger.debug("Error retrieving URL %s: %s", url, e)
            return None

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks if a URL is valid.

        :param url: The URL to check.
        :return: True if the URL is valid, False otherwise.
        """

        result = urlparse(url)
        # schema is http or https and netloc is not empty
        return all([result.scheme in ["http", "https"], result.netloc])

    def _request_headers(self):
        """
        Returns the headers to be used for the HTTP request.
        """
        return {
            "accept": "*/*",
            "User-Agent": f"haystack/LinkContentRetriever/{__version__}",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8,es;q=0.7",
            "referer": "https://www.google.com/",
        }
