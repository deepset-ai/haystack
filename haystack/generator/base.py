from abc import ABC, abstractmethod
from typing import List

from haystack import Document


class BaseGenerator(ABC):

    @abstractmethod
    def predict(self, question: str, documents: List[Document]):
        pass
