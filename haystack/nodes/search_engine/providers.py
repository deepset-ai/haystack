import json
from typing import List

import requests

from haystack import Document
from haystack.nodes.search_engine.base import SearchEngine


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
        self.params_dict = {}
        self.api_key = api_key
        self.kwargs = kwargs
        self.engine = engine

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerpAPI.
        :return: List[Document]
        """
        url = "https://serpapi.com/search"

        params = {"source": "python", "serp_api_key": self.api_key, "q": query}
        params.update(kwargs)

        if self.engine:
            params["engine"] = self.engine
        response = requests.get(url, params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)
        return [Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic_results"]]


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
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query, "gl": "us", "hl": "en", "autocorrect": True})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)
        organic = [Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic"]]
        people_also_ask = []
        if "peopleAlsoAsk" in json_result:
            people_also_ask = [
                Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["peopleAlsoAsk"]
            ]
        return organic + people_also_ask
