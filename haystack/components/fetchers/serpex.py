# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class SerpexWebSearch:
    """
    Fetches web search results from the SERPEX API.

    SERPEX provides web search results from multiple search engines including Google, Bing, DuckDuckGo, and more.
    Use it to retrieve organic search results, snippets, and metadata for search queries.

    ### Usage example

    ```python
    from haystack.components.fetchers import SerpexWebSearch

    fetcher = SerpexWebSearch(api_key="your-serpex-api-key")
    results = fetcher.run(query="What is Haystack?")

    documents = results["documents"]
    for doc in documents:
        print(f"Title: {doc.meta['title']}")
        print(f"URL: {doc.meta['url']}")
        print(f"Snippet: {doc.content}")
    ```
    """

    def __init__(
        self,
        api_key: str,
        engine: str = "google",
        num_results: int = 10,
        timeout: int = 10,
        retry_attempts: int = 2,
    ):
        """
        Initializes the SerpexWebSearch component.

        :param api_key: SERPEX API key for authentication. Get yours at https://serpex.dev
        :param engine: Search engine to use. Options: "auto", "google", "bing", "duckduckgo",
                      "brave", "yahoo", "yandex". Defaults to "google".
        :param num_results: Number of search results to return. Defaults to 10.
        :param timeout: Timeout in seconds for the API request. Defaults to 10.
        :param retry_attempts: Number of retry attempts for failed requests. Defaults to 2.
        """
        self.api_key = api_key
        self.engine = engine
        self.num_results = num_results
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Create httpx client
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

        # Define retry decorator
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        )
        def make_request(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> httpx.Response:
            response = self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response

        self._make_request = make_request

    def __del__(self):
        """
        Clean up resources when the component is deleted.

        Closes the HTTP client to prevent resource leaks.
        """
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            engine=self.engine,
            num_results=self.num_results,
            timeout=self.timeout,
            retry_attempts=self.retry_attempts,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerpexWebSearch":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        engine: Optional[str] = None,
        num_results: Optional[int] = None,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetches web search results for the given query.

        :param query: The search query string.
        :param engine: Override the default search engine. If None, uses the engine from initialization.
        :param num_results: Override the default number of results. If None, uses num_results from initialization.
        :param time_range: Time range filter for results. Options: "all", "day", "week", "month", "year".
                          Defaults to None (all time).
        :returns: Dictionary containing a list of Document objects with search results.
        """
        documents: List[Document] = []

        try:
            # Prepare request parameters
            params: Dict[str, Any] = {
                "q": query,
                "engine": engine or self.engine,
                "num": num_results or self.num_results,
                "category": "web",
            }

            if time_range:
                params["time_range"] = time_range

            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

            # Make API request
            response = self._make_request("https://api.serpex.dev/api/search", headers, params)
            data = response.json()

            # Parse search results (API returns 'results' not 'organic_results')
            if "results" in data and isinstance(data["results"], list):
                for result in data["results"]:
                    # Extract result data
                    title = result.get("title", "")
                    url = result.get("url", "")  # API uses 'url' not 'link'
                    snippet = result.get("snippet", "")
                    position = result.get("position", 0)

                    # Create Document object
                    doc = Document(
                        content=snippet,
                        meta={
                            "title": title,
                            "url": url,
                            "position": position,
                            "query": query,
                            "engine": engine or self.engine,
                        },
                    )
                    documents.append(doc)

                logger.info(
                    "Successfully fetched {count} search results for query: {query}",
                    count=len(documents),
                    query=query,
                )
            else:
                logger.warning(
                    "No results found in SERPEX API response for query: {query}",
                    query=query,
                )

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error occurred while fetching SERPEX results: {status} - {detail}",
                status=e.response.status_code,
                detail=str(e),
            )
            raise
        except httpx.RequestError as e:
            logger.error("Request error occurred while fetching SERPEX results: {error}", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error occurred while fetching SERPEX results: {error}", error=str(e))
            raise

        return {"documents": documents}
