from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union

from haystack import Document, BaseComponent


class BaseTranslator(BaseComponent):
    """
    Abstract class for a Translator component that translates either a query or a doc from language A to language B.
    """

    outgoing_edges = 1

    @abstractmethod
    def translate(
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[Document], List[str], List[Dict[str, Any]]]:
        """
        Translate the passed query or a list of documents from language A to B.
        """
        pass

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        documents: Optional[Union[List[Document], List[str], List[Dict[str, Any]]]] = None,
        answers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        dict_key: Optional[str] = None,
        **kwargs
    ):
        """Method that gets executed when this class is used as a Node in a Haystack Pipeline"""

        results: Dict = {
            **kwargs
        }

        # This will cover input query stage
        if query:
            results["query"] = self.translate(query=query)
        # This will cover retriever and summarizer
        if documents:
            dict_key = dict_key or "text"
            results["documents"] = self.translate(documents=documents, dict_key=dict_key)

        if answers:
            dict_key = dict_key or "answer"
            if isinstance(answers, Mapping):
                # This will cover reader
                results["answers"] = self.translate(documents=answers["answers"], dict_key=dict_key)
            else:
                # This will cover generator
                results["answers"] = self.translate(documents=answers, dict_key=dict_key)

        return results, "output_1"