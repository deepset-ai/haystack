from abc import ABCMeta, abstractmethod
from typing import List

from haystack.database.base import Document


class BaseRetriever(ABCMeta):

    @abstractmethod
    def retrieve(self, query: str, filters: dict = None, top_k: int = 10, index: str = None) -> List[Document]:
        pass
