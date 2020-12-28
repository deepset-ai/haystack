from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from haystack import Document


class BaseSummarizer(ABC):
    """
    Abstract class for Summarizer
    """

    outgoing_edges = 1

    @abstractmethod
    def predict(self, documents: List[Document], generate_one_summary: bool = False, query: str = None) -> Dict:
        """
        Abstract method to produce summarizer.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param generate_one_summary: To generate single summary for all documents
        :return: Summarization plus additional infos in a dict
        """
        pass

    def run(self, documents: List[Document], generate_one_summary: bool = False, query: str = None, **kwargs):

        if documents:
            results = self.predict(query=query, documents=documents, generate_one_summary=generate_one_summary)
        else:
            results = {"answers": []}

        results.update(**kwargs)
        return results, "output_1"
