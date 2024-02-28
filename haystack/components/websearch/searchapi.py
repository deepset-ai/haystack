import json
from typing import Any, Dict, List, Optional

import requests

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

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
        api_key: Secret = Secret.from_env_var("SEARCHAPI_API_KEY"),
        top_k: Optional[int] = 10,
        allowed_domains: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for the SearchApi API
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param search_params: Additional parameters passed to the SearchApi API.
        For example, you can set 'num' to 100 to increase the number of search results.
        See the [SearchApi website](https://www.searchapi.io/) for more details.
        """

        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

        # Ensure that the API key is resolved.
        _ = self.api_key.resolve_value()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            top_k=self.top_k,
            allowed_domains=self.allowed_domains,
            search_params=self.search_params,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchApiWebSearch":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document], links=List[str])
    def run(self, query: str):
        """
        Search the SearchApi API for the given query and return the results as a list of Documents and a list of links.

        :param query: Query string.
        """
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""

        payload = json.dumps({"q": query_prepend + " " + query, **self.search_params})
        headers = {"Authorization": f"Bearer {self.api_key.resolve_value()}", "X-SearchApi-Source": "Haystack"}

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

        logger.debug(
            "SearchApi returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}
