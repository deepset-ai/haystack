import json
import logging
from typing import Dict, List, Optional, Any

import requests

from haystack.preview import Document, component, default_from_dict, default_to_dict
from haystack.preview.components.websearch.utils import score_results

logger = logging.getLogger(__name__)


@component
class SerperDev:
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
        self.search_params = search_params if search_params else {}

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
    def from_dict(cls, data: Dict[str, Any]) -> "SerperDev":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search the SerperDev API for the given query and return the results as a list of Documents.

        :param query: Query string.
        :param allowed_domains: List of domains to limit the search to.
        :param top_k: Number of documents to return.
        :param search_params: Additional parameters passed to the SerperDev API.

        Refer to the [Serper Dev website](https://serper.dev/) for more details on the API parameters.
        :return: List[Document]
        """
        top_k = top_k or self.top_k
        allowed_domains = allowed_domains or self.allowed_domains
        search_params = search_params or {}
        search_params = {**self.search_params, **search_params}
        query_prepend = "OR ".join(f"site:{domain} " for domain in allowed_domains) if allowed_domains else ""

        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query_prepend + query, "gl": "us", "hl": "en", "autocorrect": True, **search_params})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            # If we reached this point, it means the request was successful and we can proceed

            json_result = response.json()

            # we get the snippet from the json result and put it in the content field of the document
            organic = [
                Document.from_dict(
                    {"metadata": {k: v for k, v in d.items() if k != "snippet"}, "content": d["snippet"]}
                )
                for d in json_result["organic"]
            ]

            answer_box = []
            # answer box is what search engine shows as a direct answer to the query
            if "answerBox" in json_result:
                answer_dict = json_result["answerBox"]
                highlighted_answers = answer_dict.get("snippetHighlighted")
                answer_box_content = (
                    highlighted_answers[0]
                    if isinstance(highlighted_answers, list) and len(highlighted_answers) > 0
                    else highlighted_answers
                    if isinstance(highlighted_answers, str)
                    else None
                )
                if not answer_box_content:
                    for key in ["snippet", "answer", "title"]:
                        if key in answer_dict:
                            answer_box_content = answer_dict[key]
                            break
                if answer_box_content:
                    answer_box = [
                        Document.from_dict(
                            {
                                "content": answer_box_content,
                                "metadata": {
                                    "title": answer_dict.get("title", ""),
                                    "link": answer_dict.get("link", ""),
                                },
                            }
                        )
                    ]

            people_also_ask = []
            # these are related questions that search engine shows
            if "peopleAlsoAsk" in json_result:
                for result in json_result["peopleAlsoAsk"]:
                    title = result.get("title", "")
                    people_also_ask.append(
                        Document.from_dict(
                            {
                                "content": result["snippet"] if result.get("snippet") else title,
                                "metadata": {"title": title, "link": result.get("link", None)},
                            }
                        )
                    )

            documents = answer_box + organic + people_also_ask

            logger.debug("Serper Dev returned %s documents for the query '%s'", len(documents), query)
            result_docs = documents[:top_k]
            return score_results(result_docs, len(answer_box) > 0)

        except requests.Timeout:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.")

        except requests.RequestException as e:
            raise Exception(f"An error occurred while querying {self.__class__.__name__}. Error: {e}")

        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
