from abc import ABC, abstractmethod
from typing import List, Optional

from haystack.schema import Document
from haystack.nodes.base import BaseComponent


class BaseQueryClassifier(BaseComponent, ABC):
    """
    Abstract class for Query Classifiers
    """

    outgoing_edges = 2

    @abstractmethod
    def run(self, documents: List[Document], generate_single_summary: Optional[bool] = None):  # type: ignore
        raise NotImplementedError()
