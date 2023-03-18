import pydoc
from typing import List, Dict, Any, Optional, Union, Tuple, Type, Callable

from haystack import BaseComponent, MultiLabel, Document
from haystack.nodes import PromptNode, PromptTemplate, TopPSampler
from haystack.nodes.search_engine.base import SearchEngine


class WebSearch(BaseComponent):
    """
    WebSearch queries a search engine and retrieves results as a list of Documents. WebSearch abstracts away the details
    of the underlying search engine provider, provides common interface for all providers and allows use of various
    search engines.

    The following search engines providers(bridges) are currently supported:
    - SerperDev (default)
    - SerpAPI
    - BingAPI

    """

    outgoing_edges = 1

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        search_engine_provider: Union[str, SearchEngine] = "SerperDev",
        **kwargs,
    ):
        """
        :param api_key: API key for the search engine provider.
        :param search_engine_provider: Name of the search engine provider class, see providers.py for a list of
        supported providers.
        :param kwargs: Additional parameters to pass to the search engine provider.
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
            self.search_engine = klass(api_key=api_key, top_k=top_k, **kwargs)  # type: ignore
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
        :param query: The query to search for

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
            raise ValueError("NeuralWebSearch run_batch requires the `queries` parameter to be Union[str, List[str]]")
        for query in queries:
            results.append(self.search_engine.search(query))
        return {"documents": results}, "output_1"


class NeuralWebSearch(BaseComponent):
    """
    NeuralWebSearch queries a search engine, retrieves the results, and uses PromptNode along with PromptTemplate
    to extract the final answer from the retrieved results, effectively building an extractive QA system on top
    of provided WebSearch component.
    """

    outgoing_edges = 1

    def __init__(
        self,
        websearch: WebSearch,
        prompt_node: PromptNode,
        prompt_template: PromptTemplate,
        prepare_template_params_fn: Callable[[List[Document], Dict[str, Any]], Dict[str, str]],
        extract_final_answer_fn: Callable[[str], str],
        top_p: float = 0.95,
        **kwargs,
    ):
        """
        :param websearch: WebSearch node.
        :param prompt_node: PromptNode node.
        :param prompt_template: The name of the PromptTemplate to use for the PromptNode.
        :param prepare_template_params_fn: Function that prepares the template parameters for the prompt template.
        :param extract_final_answer_fn: Function that extracts the final answer from the prompt node output.
        """
        super().__init__()
        self.websearch = websearch
        self.sampler = TopPSampler(top_p=top_p)
        self.prompt_node = prompt_node
        self.prompt_template = prompt_template
        self.prepare_template_params_fn = prepare_template_params_fn
        self.extract_final_answer_fn = extract_final_answer_fn

    def query_and_extract_answer(self, query: str):
        result, _ = self.websearch.run(query=query)
        doc_hits: List[Document] = result["documents"]
        doc_hits = self.sampler.predict(query=query, documents=doc_hits)
        prompt_kwargs = self.prepare_template_params_fn(doc_hits, {"query": query})
        response = self.prompt_node.prompt(self.prompt_template, **prompt_kwargs)
        final_answer = self.extract_final_answer_fn(next(iter(response)))
        return final_answer

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
        """
        if not query:
            raise ValueError("NeuralWebSearch requires the `query` parameter.")

        final_answer = self.query_and_extract_answer(query)
        return {"output": final_answer}, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ) -> Tuple[Dict, str]:
        """
        Search the search engine for the given query and return the results. Only the query parameter is used.
        """
        results = []
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            raise ValueError("NeuralWebSearch run_batch requires the `queries` parameter to be Union[str, List[str]]")
        for query in queries:
            results.append(self.query_and_extract_answer(query=query))
        return {"output": results}, "output_1"
