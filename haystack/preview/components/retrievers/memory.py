from typing import Dict, List, Any, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import MemoryDocumentStore, StoreAwareMixin


@component
class MemoryRetriever(StoreAwareMixin):
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.

    Needs to be connected to a MemoryDocumentStore to run.
    """

    supported_stores = [MemoryDocumentStore]

    @component.input
    def input(self):  # type: ignore
        class Input:
            """
            Input data for the MemoryRetriever component.

            :param query: The query string for the retriever.
            :param filters: A dictionary with filters to narrow down the search space.
            :param top_k: The maximum number of documents to return.
            :param scale_score: Whether to scale the BM25 scores or not.
            :param stores: A dictionary mapping document store names to instances.
            """

            queries: List[str]
            filters: Dict[str, Any]
            top_k: int
            scale_score: bool

        return Input

    @component.output
    def output(self):  # type: ignore
        class Output:
            """
            Output data from the MemoryRetriever component.

            :param documents: The retrieved documents.
            """

            documents: List[List[Document]]

        return Output

    def __init__(self, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = True):
        """
        Create a MemoryRetriever component.

        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param scale_score: Whether to scale the BM25 score or not (default is True).

        :raises ValueError: If the specified top_k is not > 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.defaults = {"top_k": top_k, "scale_score": scale_score, "filters": filters or {}}

    def run(self, data):
        """
        Run the MemoryRetriever on the given input data.

        :param data: The input data for the retriever.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        self.store: MemoryDocumentStore

        if not self.store:
            raise ValueError("MemoryRetriever needs a store to run: set the store instance to the self.store attribute")
        docs = []
        for query in data.queries:
            docs.append(
                self.store.bm25_retrieval(
                    query=query, filters=data.filters, top_k=data.top_k, scale_score=data.scale_score
                )
            )
        return self.output(documents=docs)
