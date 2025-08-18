# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional, Union

import requests

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

BRIGHTDATA_BASE_URL = "https://api.brightdata.com/request"


class BrightDataError(ComponentError): ...


@component
class BrightDataWebSearch:
    """
    Uses [BrightData](https://brightdata.com/) to search the web for relevant documents.

    See the [BrightData website](https://brightdata.com/) for more details.

    Usage example:
    ```python
    from haystack.components.websearch import BrightDataWebSearch
    from haystack.utils import Secret

    websearch = BrightDataWebSearch(top_k=10, api_key=Secret.from_token("test-api-key"))
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("BRIGHTDATA_API_KEY"),
        top_k: Optional[int] = 10,
        allowed_domains: Optional[list[str]] = None,
        search_params: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the BrightDataWebSearch component.

        :param api_key: API key for the BrightData API.
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param search_params: Additional parameters passed to the BrightData API.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
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
            allowed_domains=self.allowed_domains,
            search_params=self.search_params,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrightDataWebSearch":
        """
        Serializes the component to a dictionary.

        :returns:
                Dictionary with serialized data.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, Union[list[Document], list[str]]]:
        """
        Use [BrightData](https://brightdata.com/) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises BrightDataError: If an error occurs while querying the BrightData API.
        :raises TimeoutError: If the request to the BrightData API times out.
        """
        query_prepend = " OR ".join(f"site:{domain}" for domain in self.allowed_domains) if self.allowed_domains else ""

        payload = {"query": query_prepend + query, "format": "json", "search_engine": "google"}  # Default payload
        payload.update(self.search_params)  # Update with any additional  or updated search parameters

        # zone is required for BrightData API
        if payload.get("zone") is None:
            raise BrightDataError(
                "The 'zone' parameter is required for BrightData API requests. Please provide it in 'search_params'."
            )

        headers = {"Authorization": f"Bearer {self.api_key.resolve_value()}", "Content-Type": "application/json"}

        try:
            response = requests.post(BRIGHTDATA_BASE_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except requests.Timeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error

        except requests.RequestException as e:
            raise BrightDataError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        # If we reached this point, it means the request was successful and we can proceed
        json_result = response.json()

        # Extract organic results
        organic = [
            Document(meta={k: v for k, v in d.items() if k != "snippet"}, content=d.get("snippet"))
            for d in json_result.get("results", [])
        ]

        documents = organic
        links = [result["link"] for result in json_result.get("results", [])]

        logger.debug(
            "BrightData returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}
