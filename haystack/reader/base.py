from abc import ABC, abstractmethod
from typing import List, Optional

from haystack.database.base import Document


class BaseReader(ABC):

    @abstractmethod
    def predict(self, question: str, documents: List[Document], top_k: Optional[int] = None):
        pass
