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
    def predict(self, documents: List[Document], generate_single_summary: Optional[bool] = None) -> List[Document]:
        """
        Abstract method for creating a summary.

        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param generate_single_summary: This parameter is deprecated and will be removed in Haystack 1.12
        :return: List of Documents, where Document.meta["summary"] contains the summarization
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        documents: Union[List[Document], List[List[Document]]],
        generate_single_summary: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def run(self, documents: List[Document], generate_single_summary: Optional[bool] = None):  # type: ignore

        results: Dict = {"documents": []}

        if documents:
            results["documents"] = self.predict(documents=documents, generate_single_summary=generate_single_summary)

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: Union[List[Document], List[List[Document]]],
        generate_single_summary: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ):

        results = self.predict_batch(
            documents=documents, batch_size=batch_size, generate_single_summary=generate_single_summary
        )

        return {"documents": results}, "output_1"
