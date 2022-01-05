from typing import Any, Dict, List, Mapping, Optional, Union
from copy import deepcopy
from abc import abstractmethod

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, Answer


class BaseTranslator(BaseComponent):
    """
    Abstract class for a Translator component that translates either a query or a doc from language A to language B.
    """
    outgoing_edges = 1

    @abstractmethod
    def translate(
        self,
        results: List[Dict[str, Any]] = None,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ) -> Union[str, List[Document], List[Answer], List[str], List[Dict[str, Any]]]:
        """
        Translate the passed query or a list of documents from language A to B.
        """
        pass

    def run(  # type: ignore
        self,
        results: List[Dict[str, Any]] = None,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ):
        """Method that gets executed when this class is used as a Node in a Haystack Pipeline"""
        translation_results = {}
        
        if results is not None:
            translation_results = {"results":deepcopy(results)}
            translated_queries_answers = self.translate(results=translation_results["results"])
            for i, result in enumerate(translation_results["results"]):
                result["query"] = translated_queries_answers[i]
                result["answers"][0].answer = translated_queries_answers[len(translation_results["results"])+i]
            return translation_results, "output_1"

        # This will cover input query stage
        if query:
            translation_results["query"] = self.translate(query=query)  # type: ignore
        # This will cover retriever and summarizer
        if documents:
            _dict_key = dict_key or "text"
            translation_results["documents"] = self.translate(documents=documents, dict_key=_dict_key)  # type: ignore

        if answers:
            _dict_key = dict_key or "answer"
            if isinstance(answers, Mapping):
                # This will cover reader
                translation_results["answers"] = self.translate(documents=answers["answers"], dict_key=_dict_key)  # type: ignore
            else:
                # This will cover generator
                translation_results["answers"] = self.translate(documents=answers, dict_key=_dict_key)  # type: ignore

        return translation_results, "output_1"
