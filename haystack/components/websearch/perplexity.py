# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret
from haystack.version import __version__

logger = logging.getLogger(__name__)


PERPLEXITY_BASE_URL = "https://api.perplexity.ai/search"
_INTEGRATION_HEADER = f"haystack-core/{__version__}"


class PerplexityError(ComponentError): ...


@component
class PerplexityWebSearch:
    """
    Uses the [Perplexity Search API](https://docs.perplexity.ai/api-reference/search-post) to search the web.

    Usage example:
    <!-- test-ignore -->
    ```python
    from haystack.components.websearch import PerplexityWebSearch
    from haystack.utils import Secret

    websearch = PerplexityWebSearch(top_k=10, api_key=Secret.from_env_var("PERPLEXITY_API_KEY"))
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
        top_k: int | None = 10,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the PerplexityWebSearch component.

        :param api_key: API key for the Perplexity Search API.
        :param top_k: Number of documents to return. Maps to the API's `max_results` parameter (1-20).
        :param search_params: Additional parameters passed to the Perplexity Search API.
            Supported fields include `country`, `search_recency_filter`, `search_domain_filter`,
            `search_language_filter`, `last_updated_after_filter`, `last_updated_before_filter`,
            `search_after_date_filter`, `search_before_date_filter`, and `max_tokens_per_page`.
            See the [Perplexity Search API docs](https://docs.perplexity.ai/api-reference/search-post)
            for details.
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
        return default_to_dict(self, top_k=self.top_k, search_params=self.search_params, api_key=self.api_key)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerplexityWebSearch":
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
        Use the Perplexity Search API to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises PerplexityError: If an error occurs while querying the Perplexity API.
        :raises TimeoutError: If the request to the Perplexity API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            response = httpx.post(PERPLEXITY_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error

        except httpx.HTTPStatusError as e:
            raise PerplexityError(
                f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            ) from e

        except httpx.HTTPError as e:
            raise PerplexityError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        documents, links = self._parse_response(response)

        logger.debug(
            "Perplexity returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously use the Perplexity Search API to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises PerplexityError: If an error occurs while querying the Perplexity API.
        :raises TimeoutError: If the request to the Perplexity API times out.
        """
        payload, headers = self._prepare_request(query)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(PERPLEXITY_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error

        except httpx.HTTPStatusError as e:
            raise PerplexityError(
                f"An error occurred while querying {self.__class__.__name__}. Error: {e}, Response: {e.response.text}"
            ) from e

        except httpx.HTTPError as e:
            raise PerplexityError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        documents, links = self._parse_response(response)

        logger.debug(
            "Perplexity returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    def _prepare_request(self, query: str) -> tuple[dict[str, Any], dict[str, str]]:
        if (api_key := self.api_key.resolve_value()) is None:
            raise ValueError("API key cannot be `None`.")
        payload: dict[str, Any] = {"query": query}
        if self.top_k is not None:
            payload["max_results"] = self.top_k
        payload.update({k: v for k, v in self.search_params.items() if v is not None})
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Pplx-Integration": _INTEGRATION_HEADER,
        }
        return payload, headers

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[list[Document], list[str]]:
        json_result = response.json()
        results = json_result.get("results", [])

        documents: list[Document] = []
        links: list[str] = []
        for result in results:
            url = result.get("url", "")
            documents.append(
                Document(content=result.get("snippet", ""), meta={k: v for k, v in result.items() if k != "snippet"})
            )
            if url:
                links.append(url)

        return documents, links
