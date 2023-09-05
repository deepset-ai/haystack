import json
import logging
from typing import Dict, List, Optional, Any

import requests

from haystack.preview import Document, component, default_from_dict, default_to_dict
from haystack.preview.components.websearch._utils import add_scores_to_results

logger = logging.getLogger(__name__)


@component
class SerperDevSearchAPI:
    """
    Search engine using SerperDev API. See the [Serper Dev website](https://serper.dev/) for more details.
    """

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        allowed_domains: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for the SerperDev API.
        :param top_k: Number of documents to return.
        :param allowed_domains: List of domains to limit the search to.
        :param search_params: Additional parameters passed to the SerperDev API.
        For example, you can set 'num' to 20 to increase the number of search results.

        """
        if api_key is None:
            raise ValueError("API key for SerperDev API must be set.")
        self.api_key = api_key
        self.top_k = top_k
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key,
            top_k=self.top_k,
            allowed_domains=self.allowed_domains,
            search_params=self.search_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerperDevSearchAPI":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str) -> List[Document]:
        """
        Search the SerperDev API for the given query and return the results as a list of Documents.

        :param query: Query string.
        :return: List[Document]
        """
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""

        url = "https://google.serper.dev/search"

        payload = json.dumps(
            {"q": query_prepend + query, "gl": "us", "hl": "en", "autocorrect": True, **self.search_params}
        )
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
        except requests.Timeout:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.")

        except requests.RequestException as e:
            raise Exception(f"An error occurred while querying {self.__class__.__name__}. Error: {e}")

        # If we reached this point, it means the request was successful and we can proceed
        json_result = response.json()

        # we get the snippet from the json result and put it in the content field of the document
        organic = [
            Document.from_dict({"metadata": {k: v for k, v in d.items() if k != "snippet"}, "content": d["snippet"]})
            for d in json_result["organic"]
        ]

        answer_box = []
        # answer box is what search engine shows as a direct answer to the query
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
            if answer_box_content:
                answer_box = [
                    Document(
                        content=answer_box_content,
                        metadata={"title": answer_dict.get("title", ""), "link": answer_dict.get("link", "")},
                    )
                ]

        people_also_ask = []
        # these are related questions that search engine shows
        if "peopleAlsoAsk" in json_result:
            for result in json_result["peopleAlsoAsk"]:
                title = result.get("title", "")
                people_also_ask.append(
                    Document(
                        content=result["snippet"] if result.get("snippet") else title,
                        metadata={"title": title, "link": result.get("link", None)},
                    )
                )

        documents = answer_box + organic + people_also_ask

        logger.debug("Serper Dev returned %s documents for the query '%s'", len(documents), query)
        result_docs = documents[: self.top_k]
        return add_scores_to_results(result_docs, len(answer_box) > 0)
