from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from haystack import Document


class BaseGenerator(ABC):
    """
    Abstract class for Generators
    """
    @abstractmethod
    def predict(self, question: str, documents: List[Document], top_k: Optional[int]) -> Dict:
        """
        Abstract method to generate answers.

        :param question: Question
        :param documents: Related documents (e.g. coming from a retriever) that the answer shall be conditioned on.
        :param top_k: Number of returned answers
        :return: Generated answers plus additional infos in a dict
        """
        pass
