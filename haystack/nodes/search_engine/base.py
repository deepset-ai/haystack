from abc import ABC, abstractmethod
from typing import List

from haystack import Document


class SearchEngine(ABC):
    """
    Abstract base class for search engines providers.
    """

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Document]:
        """
        Search the search engine for the given query and return the results.
        :param query: The query to search for.
        :param kwargs: Additional parameters to pass to the search engine.
        :return: List of search results as documents.
        """
