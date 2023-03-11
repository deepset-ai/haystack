import json
import logging
from typing import Dict, List, Union

import requests

from haystack import Document
from haystack.nodes.search_engine.base import SearchEngine

logger = logging.getLogger(__name__)


class SerpAPI(SearchEngine):
    """
    SerpAPI is a search engine that provides a REST API to access search results from Google, Bing, Yahoo, Yandex,
    Amazon etc. See https://serpapi.com/ for more details.
    """

    def __init__(self, api_key: str, engine: str = "google", **kwargs):
        """
        :param api_key: API key for SerpAPI.
        :param engine: Search engine to use.
        :param kwargs: Additional parameters passed to the SerperDev API.
        """
        super().__init__()
        self.params_dict: Dict[str, Union[str, int, float]] = {}
        self.api_key = api_key
        self.kwargs = kwargs
        self.engine = engine

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerpAPI.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}

        url = "https://serpapi.com/search"

        params = {"source": "python", "serp_api_key": self.api_key, "q": query, **kwargs}

        if self.engine:
            params["engine"] = self.engine
        response = requests.get(url, params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)

        organic = [Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic_results"]]

        people_also_search = []
        if "people_also_search_for" in json_result:
            for result in json_result["people_also_search_for"]:
                people_also_search.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        related_questions = []
        if "related_questions" in json_result:
            for result in json_result["related_questions"]:
                related_questions.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        documents = organic + people_also_search + related_questions

        logger.debug("SerpAPI returned %s documents for the query '%s'", len(documents), query)
        return documents


class SerperDev(SearchEngine):
    """
    Search engine using SerperDev API. See https://serper.dev/ for more details.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        :param api_key: API key for SerperDev API.
        :param kwargs: Additional parameters passed to the SerperDev API.
        """
        super().__init__()
        self.api_key = api_key
        self.kwargs = kwargs

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerperDev API.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}

        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query, "gl": "us", "hl": "en", "autocorrect": True, **kwargs})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = response.json()
        organic = [Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic"]]
        people_also_ask = []
        if "peopleAlsoAsk" in json_result:
            for result in json_result["peopleAlsoAsk"]:
                people_also_ask.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        documents = organic + people_also_ask

        logger.debug("Serper Dev returned %s documents for the query '%s'", len(documents), query)
        return documents


class BingAPI(SearchEngine):
    """
    Search engine using Bing API. See https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview for more details.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        :param api_key: API key for Bing API.
        :param kwargs: Additional parameters passed to the SerperDev API.
        """
        super().__init__()
        self.api_key = api_key
        self.kwargs = kwargs

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerperDev API.
                       As an example you can pass the market parameter to specify the market to use for the query. 'mkt':'en-US'
                       If you don't specify the market parameter, the default market for the user's location is used.
                       For a complete list of the market codes see https://learn.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#market-codes
                       You can also pass the count parameter to specify the number of results to return. 'count':10
                       You can find the full list of parameters here: https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#query-parameters
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        url = "https://api.bing.microsoft.com/v7.0/search"

        params: Dict[str, Union[str, int, float]] = {"q": query, "count": 50, **kwargs}

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = response.json()

        documents: List[Document] = []
        for web_page in json_result["webPages"]["value"]:
            documents.append(
                Document.from_dict(
                    {
                        "title": web_page["name"],
                        "content": web_page["snippet"],
                        "position": int(web_page["id"].replace("https://api.bing.microsoft.com/api/v7/#WebPages.", "")),
                        "link": web_page["url"],
                        "language": web_page["language"],
                    }
                )
            )
            if web_page.get("deepLinks"):
                for deep_link in web_page["deepLinks"]:
                    documents.append(
                        Document.from_dict(
                            {
                                "title": deep_link["name"],
                                "content": deep_link["snippet"] if deep_link.get("snippet") else deep_link["name"],
                                "link": deep_link["url"],
                            }
                        )
                    )

        logger.debug("Bing API returned %s documents for the query '%s'", len(documents), query)
        return documents
