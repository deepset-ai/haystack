# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class SerpexWebSearch:
    """
    Fetches web search results from the SERPEX API.

    SERPEX provides web search results from multiple search engines including Google, Bing, DuckDuckGo, and more.
    Use it to retrieve organic search results, snippets, and metadata for search queries.

    ### Usage example

    ```python
    from haystack.components.websearch import SerpexWebSearch
    from haystack.utils import Secret

    fetcher = SerpexWebSearch(api_key=Secret.from_token("your-serpex-api-key"))
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
        *,
        api_key: Secret = Secret.from_env_var("SERPEX_API_KEY"),
        engine: Literal["auto", "google", "bing", "duckduckgo", "brave", "yahoo", "yandex"] = "google",
        num_results: int = 10,
        timeout: float = 10.0,
        retry_attempts: int = 2,
    ) -> None:
        """
        Initializes the SerpexWebSearch component.

        :param api_key: SERPEX API key for authentication. Get yours at https://serpex.dev
        :param engine: Search engine to use. Options: "auto", "google", "bing", "duckduckgo",
                      "brave", "yahoo", "yandex". Defaults to "google".
        :param num_results: Number of search results to return. Defaults to 10.
        :param timeout: Timeout in seconds for the API request. Defaults to 10.0.
        :param retry_attempts: Number of retry attempts for failed requests. Defaults to 2.
        """
        self.api_key = api_key
        self.engine = engine
        self.num_results = num_results
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Ensure that the API key is resolved
        _ = self.api_key.resolve_value()

        # Create httpx client
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

        # Define retry decorator
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        )
        def make_request(url: str, headers: dict[str, str], params: dict[str, Any]) -> httpx.Response:
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            engine=self.engine,
            num_results=self.num_results,
            timeout=self.timeout,
            retry_attempts=self.retry_attempts,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerpexWebSearch":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        engine: Optional[Literal["auto", "google", "bing", "duckduckgo", "brave", "yahoo", "yandex"]] = None,
        num_results: Optional[int] = None,
        time_range: Optional[str] = None,
    ) -> dict[str, list[Document]]:
        """
        Fetches web search results for the given query.

        :param query: The search query string.
        :param engine: Override the default search engine. If None, uses the engine from initialization.
        :param num_results: Override the default number of results. If None, uses num_results from initialization.
        :param time_range: Time range filter for results. Options: "all", "day", "week", "month", "year".
                          Defaults to None (all time).
        :returns: Dictionary containing a list of Document objects with search results.
        """
        documents: list[Document] = []

        try:
            # Prepare request parameters
            params: dict[str, Any] = {
                "q": query,
                "engine": engine or self.engine,
                "num": num_results or self.num_results,
                "category": "web",
            }

            if time_range:
                params["time_range"] = time_range

            headers = {"Authorization": f"Bearer {self.api_key.resolve_value()}", "Content-Type": "application/json"}

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
