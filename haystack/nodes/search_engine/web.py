import pydoc
from typing import List, Dict, Any, Optional, Union, Tuple, Type, Callable

from haystack import BaseComponent, MultiLabel, Document
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes.search_engine.base import SearchEngine


class WebSearch(BaseComponent):
    """
    WebSearch queries a search engine and retrieve results. It can use various search engine providers, e.g. SerperDev,
    SerpAPI, etc.
    """

    outgoing_edges = 1

    def __init__(self, api_key: str, search_engine_provider: Union[str, SearchEngine] = "SerperDev", **kwargs):
        """
        :param api_key: API key for the search engine provider
        :param search_engine_provider: Name of the search engine provider, e.g. "SerperDev", "SerpAPI"
        :param kwargs: Additional parameters to pass to the search engine provider
        """
        super().__init__()
        if isinstance(search_engine_provider, str):
            # try to find the provider class
            search_path = [f"haystack.nodes.search_engine.providers.{search_engine_provider}", search_engine_provider]
            klass: Type[SearchEngine] = next((pydoc.locate(path) for path in search_path), None)  # type: ignore

            if not klass:
                raise ValueError(
                    f"Could not locate SearchEngine class with name {search_engine_provider}. "
                    f"Make sure to pass the full path to the class."
                )
            if not issubclass(klass, SearchEngine):
                raise ValueError(f"Class {search_engine_provider} is not a subclass of SearchEngine.")
            self.search_engine = klass(api_key=api_key, **kwargs)  # type: ignore
        elif isinstance(search_engine_provider, SearchEngine):
            self.search_engine = search_engine_provider
        else:
            raise ValueError(
                "search_engine_provider must be either a string (SearchEngine class name) or a SearchEngine instance"
            )

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
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
        pass


class NeuralWebSearch(BaseComponent):
    """
    NeuralWebSearch queries a search engine, retrieve results, and uses PromptNode along with PromptTemplate
    to extract the final answer from the retrieved results effectively building a QA system on top of a search engine.
    """

    outgoing_edges = 1

    def __init__(
        self,
        websearch: WebSearch,
        prompt_node: PromptNode,
        prompt_template: PromptTemplate,
        prepare_template_params_fn: Callable[[List[Document], Dict[str, Any]], Dict[str, str]],
        extract_final_answer_fn: Callable[[str], str],
        **kwargs,
    ):
        """
        :param websearch: WebSearch node
        :param prompt_node: PromptNode node
        :param prompt_template: PromptTemplate to use for the prompt node
        :param prepare_template_params_fn: Function that prepares the template parameters for the prompt template
        :param extract_final_answer_fn: Function that extracts the final answer from the prompt node output
        """
        super().__init__()
        self.websearch = websearch
        self.prompt_node = prompt_node
        self.prompt_template = prompt_template
        self.prepare_template_params_fn = prepare_template_params_fn
        self.extract_final_answer_fn = extract_final_answer_fn

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        result, _ = self.websearch.run(query=query)

        doc_hits: List[Document] = result["output"]
        prompt_kwargs = self.prepare_template_params_fn(doc_hits, {"query": query, "documents": documents})

        response = self.prompt_node.prompt(self.prompt_template, **prompt_kwargs)
        final_answer = self.extract_final_answer_fn(next(iter(response)))

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
    ):
        pass
