import json
import logging
from typing import Dict, List, Union, Optional, Any

import requests

from haystack import Document
from haystack.nodes.search_engine.base import SearchEngine

logger = logging.getLogger(__name__)


class SerpAPI(SearchEngine):
    """
    SerpAPI is a search engine that provides a REST API to access search results from Google, Bing, Yahoo, Yandex,
    Amazon, and similar. See the [SerpAPI website](https://serpapi.com/) for more details.
    """

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        engine: Optional[str] = "google",
        search_engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for SerpAPI.
        :param top_k: Number of results to return.
        :param engine: Search engine to use, for example google, bing, baidu, duckduckgo, yahoo, yandex.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported engines.
        :param search_engine_kwargs: Additional parameters passed to the SerperDev API. For example, you can set 'lr' to 'lang_en'
        to limit the search to English.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported parameters.
        """
        super().__init__()
        self.params_dict: Dict[str, Union[str, int, float]] = {}
        self.api_key = api_key
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}
        self.engine = engine
        self.top_k = top_k

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerpAPI. For example, you can set 'lr' to 'lang_en'
        to limit the search to English.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported parameters.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        top_k = kwargs.pop("top_k", self.top_k)
        url = "https://serpapi.com/search"

        params = {"source": "python", "serp_api_key": self.api_key, "q": query, **kwargs}

        if self.engine:
            params["engine"] = self.engine
        response = requests.get(url, params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)

        organic = [
            Document.from_dict(d, field_map={"snippet": "content"})
            for d in json_result["organic_results"]
            if "snippet" in d
        ]
        answer_box = []
        if "answer_box" in json_result:
            answer_dict = json_result["answer_box"]
            for key in ["answer", "snippet_highlighted_words", "snippet", "title"]:
                if key in answer_dict:
                    answer_box_content = answer_dict[key]
                    if isinstance(answer_box_content, list):
                        answer_box_content = answer_box_content[0]
                    answer_box = [
                        Document.from_dict(
                            {
                                "title": answer_dict.get("title", ""),
                                "content": answer_box_content,
                                "link": answer_dict.get("displayed_link", ""),
                            }
                        )
                    ]
                    break

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

        documents = answer_box + organic + people_also_search + related_questions

        logger.debug("SerpAPI returned %s documents for the query '%s'", len(documents), query)
        result_docs = documents[:top_k]
        return self.score_results(result_docs, len(answer_box) > 0)


class SerperDev(SearchEngine):
    """
    Search engine using SerperDev API. See the [Serper Dev website](https://serper.dev/) for more details.
    """

    def __init__(self, api_key: str, top_k: Optional[int] = 10, search_engine_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param api_key: API key for the SerperDev API.
        :param top_k: Number of documents to return.
        :param search_engine_kwargs: Additional parameters passed to the SerperDev API.
        For example, you can set 'num' to 20 to increase the number of search results.
        """
        super().__init__()
        self.api_key = api_key
        self.top_k = top_k
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerperDev API, such as top_k.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        top_k = kwargs.pop("top_k", self.top_k)

        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": query, "gl": "us", "hl": "en", "autocorrect": True, **kwargs})
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = response.json()
        organic = [
            Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic"] if "snippet" in d
        ]
        answer_box = []
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
                            "title": answer_dict.get("title", ""),
                            "content": answer_box_content,
                            "link": answer_dict.get("link", ""),
                        }
                    )
                ]

        people_also_ask = []
        if "peopleAlsoAsk" in json_result:
            for result in json_result["peopleAlsoAsk"]:
                title = result.get("title", "")
                people_also_ask.append(
                    Document.from_dict(
                        {
                            "title": title,
                            "content": result["snippet"] if result.get("snippet") else title,
                            "link": result.get("link", None),
                        }
                    )
                )

        documents = answer_box + organic + people_also_ask

        logger.debug("Serper Dev returned %s documents for the query '%s'", len(documents), query)
        result_docs = documents[:top_k]
        return self.score_results(result_docs, len(answer_box) > 0)


class BingAPI(SearchEngine):
    """
    Search engine using the Bing API. See [Bing Web Search API](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview) for more details.
    """

    def __init__(self, api_key: str, top_k: Optional[int] = 10, search_engine_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param api_key: API key for the Bing API.
        :param top_k: Number of documents to return.
        :param search_engine_kwargs: Additional parameters passed to the SerperDev API. As an example, you can pass the market parameter to specify the market to use for the query: 'mkt':'en-US'.
        """
        super().__init__()
        self.api_key = api_key
        self.top_k = top_k
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerperDev API.
                       As an example, you can pass the market parameter to specify the market to use for the query: 'mkt':'en-US'.
                       If you don't specify the market parameter, the default market for the user's location is used.
                       For a complete list of the market codes, see [Market Codes](https://learn.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#market-codes).
                       You can also pass the count parameter to specify the number of results to return: 'count':10.
                       You can find a full list of parameters at [Query Parameters](https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference#query-parameters).
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        top_k = kwargs.pop("top_k", self.top_k)
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
        return documents[:top_k]


class GoogleAPI(SearchEngine):
    """Search engine using the Google API. See [Google Search API](https://developers.google.com/custom-search/v1/overview) for more details."""

    def __init__(
        self,
        top_k: Optional[int] = 10,
        api_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        search_engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param top_k: Number of documents to return.
        :param api_key: API key for the Google API.
        :param engine_id: Engine ID for the Google API.
        :param search_engine_kwargs: Additional parameters passed to the Google API. As an example, you can pass the hl parameter to specify the language to use for the query: 'hl':'en'.
        """
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.top_k = top_k
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}

    def _validate_environment(self):
        """
        Validate if the environment variables are set.
        """
        if not self.api_key:
            raise ValueError(
                "You need to provide an API key for the Google API. See https://developers.google.com/custom-search/v1/overview"
            )
        if not self.engine_id:
            raise ValueError(
                "You need to provide an engine ID for the Google API. See https://developers.google.com/custom-search/v1/overview"
            )

        # check if google api is installed
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "You need to install the Google API client. You can do so by running 'pip install google-api-python-client'."
            )
        # create a custom search service
        self.service = build("customsearch", "v1", developerKey=self.api_key)

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the Google API.
                       As an example, you can pass the hl parameter to specify the language to use for the query: 'hl':'en'.
                       If you don't specify the hl parameter, the default language for the user's location is used.
                       For a complete list of the language codes, see [Language Codes](https://developers.google.com/custom-search/docs/xml_results#languageCollections).
                       You can also pass the num parameter to specify the number of results to return: 'num':10.
                       You can find a full list of parameters at [Query Parameters](https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list).
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        self.engine_id = kwargs.pop("engine_id", self.engine_id)

        self._validate_environment()

        top_k = kwargs.pop("top_k", self.top_k)
        params: Dict[str, Union[str, int, float]] = {"num": 10, **kwargs}
        res = self.service.cse().list(q=query, cx=self.engine_id, **params).execute()
        documents: List[Document] = []
        for i, result in enumerate(res["items"]):
            documents.append(
                Document.from_dict(
                    {"title": result["title"], "content": result["snippet"], "position": i, "link": result["link"]}
                )
            )
        logger.debug("Google API returned %s documents for the query '%s'", len(documents), query)
        return documents[:top_k]
