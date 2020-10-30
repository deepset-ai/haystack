from abc import ABC, abstractmethod
from typing import List, Optional

from haystack import Document


class BaseGenerator(ABC):

    @abstractmethod
    def predict(self, question: str, documents: List[Document], top_k: Optional[int]):
        pass
