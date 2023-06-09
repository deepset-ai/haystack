import pydoc
from typing import List, Dict, Any, Optional, Union, Tuple, Type

from haystack.nodes.base import BaseComponent
from haystack import MultiLabel, Document
from haystack.nodes.search_engine.base import SearchEngine


class WebSearch(BaseComponent):
    """
    WebSearch queries a search engine and retrieves results as a list of Documents. WebSearch abstracts away the details
    of the underlying search engine provider, provides common interface for all providers, and makes it possible to use various
    search engines.

    WebSearch currently supports the following search engines providers (bridges):
    - SerperDev (default)
    - SerpAPI
    - BingAPI
    - GoogleAPI

    """

    outgoing_edges = 1

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        search_engine_provider: Union[str, SearchEngine] = "SerperDev",
        search_engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for the search engine provider.
        :param search_engine_provider: Name of the search engine provider class, see `providers.py` for a list of
        supported providers.
        :param search_engine_kwargs: Additional parameters to pass to the search engine provider.
        """
        super().__init__()
        if isinstance(search_engine_provider, str):
            # try to find the provider class
            search_path = [f"haystack.nodes.search_engine.providers.{search_engine_provider}", search_engine_provider]
            klass: Type[SearchEngine] = next((pydoc.locate(path) for path in search_path), None)  # type: ignore

            if not klass:
                raise ValueError(
                    f"Could not locate the SearchEngine class with the name {search_engine_provider}. "
                    f"Make sure you pass the full path to the class."
                )
            if not issubclass(klass, SearchEngine):
                raise ValueError(f"Class {search_engine_provider} is not a subclass of SearchEngine.")
            self.search_engine = klass(api_key=api_key, top_k=top_k, search_engine_kwargs=search_engine_kwargs)  # type: ignore
        elif isinstance(search_engine_provider, SearchEngine):
            self.search_engine = search_engine_provider
        else:
            raise ValueError(
                "search_engine_provider must be either a string (SearchEngine class name) or a SearchEngine instance."
            )

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        """
        Search the search engine for the given query and return the results. Only the query parameter is used.
        :param query: The query to search for.

        :return: List of search results as documents.
        """
        # query is a required parameter for search, we need to keep the signature of run() the same as in other nodes
        if not query:
            raise ValueError("WebSearch run requires the `query` parameter")
        return {"documents": self.search_engine.search(query)}, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        results = []
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            raise ValueError("WebSearch run_batch requires the `queries` parameter to be Union[str, List[str]]")
        for query in queries:
            results.append(self.search_engine.search(query))
        return {"documents": results}, "output_1"
