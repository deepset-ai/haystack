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
        :param generate_single_summary: Whether to generate a single summary for all documents or one summary per document.
                                        If set to "True", all docs will be joined to a single string that will then
                                        be summarized.
                                        Important: The summary will depend on the order of the supplied documents!
        :return: List of Documents, where Document.text contains the summarization and Document.meta["context"]
                 the original, not summarized text
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
            documents=documents, generate_single_summary=generate_single_summary, batch_size=batch_size
        )

        return {"documents": results}, "output_1"
