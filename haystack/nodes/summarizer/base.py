from typing import List, Dict, Optional, Union

from abc import abstractmethod

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


class BaseSummarizer(BaseComponent):
    """
    Abstract class for Summarizer
    """

    outgoing_edges = 1

    @abstractmethod
    def predict(self, documents: List[Document]) -> List[Document]:
        """
        Abstract method for creating a summary.

        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :return: List of Documents, where Document.meta["summary"] contains the summarization
        """
        pass

    @abstractmethod
    def predict_batch(
        self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def run(self, documents: List[Document]):  # type: ignore
        results: Dict = {"documents": []}

        if documents:
            results["documents"] = self.predict(documents=documents)

        return results, "output_1"

    def run_batch(  # type: ignore
        self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None
    ):
        results = self.predict_batch(documents=documents, batch_size=batch_size)

        return {"documents": results}, "output_1"
