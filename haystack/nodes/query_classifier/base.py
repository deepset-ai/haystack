from abc import abstractmethod
from typing import List, Optional

from haystack.nodes.base import BaseComponent


class BaseQueryClassifier(BaseComponent):
    """
    Abstract class for Query Classifiers
    """

    outgoing_edges = 2

    @abstractmethod
    def run(self, query: str):  # type: ignore
        pass

    @abstractmethod
    def run_batch(self, queries: List[str], batch_size: Optional[int] = None):  # type: ignore
        pass
