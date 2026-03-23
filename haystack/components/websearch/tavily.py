# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


TAVILY_BASE_URL = "https://api.tavily.com/search"


class TavilyError(ComponentError): ...


@component
class TavilyWebSearch:
    """
    Uses [Tavily](https://tavily.com/) to search the web for relevant documents.

    Usage example:
    ```python
    from haystack.components.websearch import TavilyWebSearch
    from haystack.utils import Secret

    websearch = TavilyWebSearch(top_k=10, api_key=Secret.from_token("test-api-key"))
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("TAVILY_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the TavilyWebSearch component.

        :param api_key: API key for the Tavily API.
        :param top_k: Number of documents to return.
        :param search_params: Additional parameters passed to the Tavily API.
            For example, you can set 'search_depth' to 'advanced' for more thorough results,
            or 'topic' to 'news' for news-focused searches.
            See the [Tavily website](https://tavily.com/) for more details.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.search_params = search_params or {}

        # Ensure that the API key is resolved.
        _ = self.api_key.resolve_value()

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            top_k=self.top_k,
            search_params=self.search_params,
            api_key=self.api_key,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TavilyWebSearch":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Uses [Tavily](https://tavily.com/) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the Tavily API times out.
        :raises TavilyError: If an error occurs while querying the Tavily API.
        """
        payload, headers = self._prepare_request(query)
        try:
            response = httpx.post(TAVILY_BASE_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error
        except httpx.HTTPError as e:
            raise TavilyError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        documents, links = self._parse_response(response)

        logger.debug(
            "Tavily returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously uses [Tavily](https://tavily.com/) to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the Tavily API times out.
        :raises TavilyError: If an error occurs while querying the Tavily API.
        """
        payload, headers = self._prepare_request(query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(TAVILY_BASE_URL, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error
        except httpx.HTTPError as e:
            raise TavilyError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        documents, links = self._parse_response(response)

        logger.debug(
            "Tavily returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    def _prepare_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        api_key = self.api_key.resolve_value()
        if api_key is None:
            raise ValueError("API key cannot be `None`.")
        payload: dict[str, Any] = {"query": query, "api_key": api_key, **self.search_params}
        if "max_results" not in payload:
            payload["max_results"] = self.top_k or 10
        headers = {"Content-Type": "application/json"}
        return payload, headers

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[list[Document], list[str]]:
        json_result = response.json()

        documents = []
        links = []
        for result in json_result.get("results", []):
            doc = Document(
                content=result.get("content", ""),
                meta={"title": result.get("title", ""), "link": result.get("url", ""), "score": result.get("score")},
            )
            documents.append(doc)
            if result.get("url"):
                links.append(result["url"])

        return documents, links
