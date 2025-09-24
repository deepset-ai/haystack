# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional, Union
from urllib.parse import urlparse

import requests

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


SERPERDEV_BASE_URL = "https://google.serper.dev/search"


class SerperDevError(ComponentError): ...


@component
class SerperDevWebSearch:
    """
    Uses [Serper](https://serper.dev/) to search the web for relevant documents.

    See the [Serper Dev website](https://serper.dev/) for more details.

    Usage example:
    ```python
    from haystack.components.websearch import SerperDevWebSearch
    from haystack.utils import Secret

    websearch = SerperDevWebSearch(top_k=10, api_key=Secret.from_token("test-api-key"))
    results = websearch.run(query="Who is the boyfriend of Olivia Wilde?")

    assert results["documents"]
    assert results["links"]

    # Example with domain filtering - exclude subdomains
    websearch_filtered = SerperDevWebSearch(
        top_k=10,
        allowed_domains=["example.com"],
        exclude_subdomains=True,  # Only results from example.com, not blog.example.com
        api_key=Secret.from_token("test-api-key")
    )
    results_filtered = websearch_filtered.run(query="search query")
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SERPERDEV_API_KEY"),
        top_k: Optional[int] = 10,
        allowed_domains: Optional[list[str]] = None,
        search_params: Optional[dict[str, Any]] = None,
        *,
        exclude_subdomains: bool = False,
    ):
        """
        Initialize the SerperDevWebSearch component.

        :param api_key: API key for the Serper API.
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param exclude_subdomains: Whether to exclude subdomains when filtering by allowed_domains.
            If True, only results from the exact domains in allowed_domains will be returned.
            If False, results from subdomains will also be included. Defaults to False.
        :param search_params: Additional parameters passed to the Serper API.
            For example, you can set 'num' to 20 to increase the number of search results.
            See the [Serper website](https://serper.dev/) for more details.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.exclude_subdomains = exclude_subdomains
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
            exclude_subdomains=self.exclude_subdomains,
            search_params=self.search_params,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerperDevWebSearch":
        """
        Serializes the component to a dictionary.

        :returns:
                Dictionary with serialized data.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _is_domain_allowed(self, url: str) -> bool:
        """
        Check if a URL's domain is allowed based on allowed_domains and exclude_subdomains settings.

        :param url: The URL to check.
        :returns: True if the domain is allowed, False otherwise.
        """
        if not self.allowed_domains:
            return True

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            for allowed_domain in self.allowed_domains:
                allowed_domain = allowed_domain.lower()

                if self.exclude_subdomains:
                    # Exact domain match only
                    if domain == allowed_domain:
                        return True
                # Allow subdomains (current behavior)
                elif domain == allowed_domain or domain.endswith("." + allowed_domain):
                    return True

            return False
        except Exception:
            # If URL parsing fails, allow the result to be safe
            return True

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, Union[list[Document], list[str]]]:
        """
        Use [Serper](https://serper.dev/) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises SerperDevError: If an error occurs while querying the SerperDev API.
        :raises TimeoutError: If the request to the SerperDev API times out.
        """
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""

        payload = json.dumps(
            {"q": query_prepend + query, "gl": "us", "hl": "en", "autocorrect": True, **self.search_params}
        )
        headers = {"X-API-KEY": self.api_key.resolve_value(), "Content-Type": "application/json"}

        try:
            response = requests.post(SERPERDEV_BASE_URL, headers=headers, data=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except requests.Timeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error

        except requests.RequestException as e:
            raise SerperDevError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e

        # If we reached this point, it means the request was successful and we can proceed
        json_result = response.json()

        # we get the snippet from the json result and put it in the content field of the document
        organic = [
            Document(meta={k: v for k, v in d.items() if k != "snippet"}, content=d.get("snippet"))
            for d in json_result["organic"]
            if self._is_domain_allowed(d.get("link", ""))
        ]

        # answer box is what search engine shows as a direct answer to the query
        answer_box = []
        if "answerBox" in json_result:
            answer_dict = json_result["answerBox"]
            highlighted_answers = answer_dict.get("snippetHighlighted")
            answer_box_content = None
            # Check if highlighted_answers is a list and has at least one element
            if isinstance(highlighted_answers, list) and len(highlighted_answers) > 0:
                answer_box_content = highlighted_answers[0]
            elif isinstance(highlighted_answers, str):
                answer_box_content = highlighted_answers
            if not answer_box_content:
                for key in ["snippet", "answer", "title"]:
                    if key in answer_dict:
                        answer_box_content = answer_dict[key]
                        break
            if answer_box_content and self._is_domain_allowed(answer_dict.get("link", "")):
                answer_box = [
                    Document(
                        content=answer_box_content,
                        meta={"title": answer_dict.get("title", ""), "link": answer_dict.get("link", "")},
                    )
                ]

        # these are related questions that search engine shows
        people_also_ask = []
        if "peopleAlsoAsk" in json_result:
            for result in json_result["peopleAlsoAsk"]:
                if self._is_domain_allowed(result.get("link", "")):
                    title = result.get("title", "")
                    people_also_ask.append(
                        Document(
                            content=result["snippet"] if result.get("snippet") else title,
                            meta={"title": title, "link": result.get("link", None)},
                        )
                    )

        documents = answer_box + organic + people_also_ask

        links = [result["link"] for result in json_result["organic"] if self._is_domain_allowed(result.get("link", ""))]

        logger.debug(
            "Serper Dev returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}
