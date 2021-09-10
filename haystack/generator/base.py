from abc import abstractmethod
from typing import List, Optional, Dict

from haystack import Document, BaseComponent


class BaseGenerator(BaseComponent):
    """
    Abstract class for Generators
    """

    outgoing_edges = 1

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int]) -> Dict:
        """
        Abstract method to generate answers.

        :param query: Query
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers plus additional infos in a dict
        """
        pass

    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None): # type: ignore

        if documents:
            results = self.predict(query=query, documents=documents, top_k=top_k)
        else:
            results = {"answers": []}

        return results, "output_1"
