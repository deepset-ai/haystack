from typing import Any, Dict, List, Mapping, Optional, Union

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
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[Answer], List[str], List[Dict[str, Any]]]] = None,
        answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
    ):
        """Method that gets executed when this class is used as a Node in a Haystack Pipeline"""

        results = {}

        # This will cover input query stage
        if query:
            results["query"] = self.translate(query=query)
        # This will cover retriever and summarizer
        if documents:
            _dict_key = dict_key or "text"
            results["documents"] = self.translate(documents=documents, dict_key=_dict_key)

        if answers:
            _dict_key = dict_key or "answer"
            if isinstance(answers, Mapping):
                # This will cover reader
                results["answers"] = self.translate(documents=answers["answers"], dict_key=_dict_key)
            else:
                # This will cover generator
                results["answers"] = self.translate(documents=answers, dict_key=_dict_key)

        return results, "output_1"