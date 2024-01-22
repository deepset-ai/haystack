import json
import logging
from typing import Dict, List, Optional, Any
import os

import requests

from haystack import Document, component, default_to_dict, ComponentError

logger = logging.getLogger(__name__)


SEARCHAPI_BASE_URL = "https://www.searchapi.io/api/v1/search"


class SearchApiError(ComponentError):
    ...


@component
class SearchApiWebSearch:
    """
    Search engine using SearchApi API. Given a query, it returns a list of URLs that are the most relevant.

    See the [SearchApi website](https://www.searchapi.io/) for more details.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: Optional[int] = 10,
        allowed_domains: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for the SearchApi API.  It can be
        explicitly provided or automatically read from the
        environment variable SEARCHAPI_API_KEY (recommended).
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param search_params: Additional parameters passed to the SearchApi API.
        For example, you can set 'num' to 100 to increase the number of search results.
        See the [SearchApi website](https://www.searchapi.io/) for more details.
        """
        api_key = api_key or os.environ.get("SEARCHAPI_API_KEY")
        # we check whether api_key is None or an empty string
        if not api_key:
            msg = (
                "SearchApiWebSearch expects an API key. "
                "Set the SEARCHAPI_API_KEY environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)

        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self, top_k=self.top_k, allowed_domains=self.allowed_domains, search_params=self.search_params
        )

    @component.output_types(documents=List[Document], links=List[str])
    def run(self, query: str):
        """
        Search the SearchApi API for the given query and return the results as a list of Documents and a list of links.

        :param query: Query string.
        """
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""

        payload = json.dumps({"q": query_prepend + " " + query, **self.search_params})
        headers = {"Authorization": f"Bearer {self.api_key}", "X-SearchApi-Source": "Haystack"}

        try:
            response = requests.get(SEARCHAPI_BASE_URL, headers=headers, params=payload, timeout=90)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except requests.Timeout:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.")

        except requests.RequestException as e:
            raise SearchApiError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        # Request succeeded
        json_result = response.json()

        # organic results are the main results from the search engine
        organic_results = []
        if "organic_results" in json_result:
            for result in json_result["organic_results"]:
                organic_results.append(
                    Document.from_dict({"title": result["title"], "content": result["snippet"], "link": result["link"]})
                )

        # answer box has a direct answer to the query
        answer_box = []
        if "answer_box" in json_result:
            answer_box = [
                Document.from_dict(
                    {
                        "title": json_result["answer_box"].get("title", ""),
                        "content": json_result["answer_box"].get("answer", ""),
                        "link": json_result["answer_box"].get("link", ""),
                    }
                )
            ]

        knowledge_graph = []
        if "knowledge_graph" in json_result:
            knowledge_graph = [
                Document.from_dict(
                    {
                        "title": json_result["knowledge_graph"].get("title", ""),
                        "content": json_result["knowledge_graph"].get("description", ""),
                    }
                )
            ]

        related_questions = []
        if "related_questions" in json_result:
            for result in json_result["related_questions"]:
                related_questions.append(
                    Document.from_dict(
                        {
                            "title": result["question"],
                            "content": result["answer"] if result.get("answer") else result.get("answer_highlight", ""),
                            "link": result.get("source", {}).get("link", ""),
                        }
                    )
                )

        documents = answer_box + knowledge_graph + organic_results + related_questions

        links = [result["link"] for result in json_result["organic_results"]]

        logger.debug("SearchApi returned %s documents for the query '%s'", len(documents), query)
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}
