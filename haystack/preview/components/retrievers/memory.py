from typing import Dict, List, Any, Optional

from haystack.preview import component, Document
from haystack.preview.document_stores import MemoryDocumentStore


@component
class MemoryRetriever:
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.

    Needs to be connected to a MemoryDocumentStore to run.
    """

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
    ):
        """
        Create a MemoryRetriever component.

        :param document_store: An instance of MemoryDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param scale_score: Whether to scale the BM25 score or not (default is True).

        :raises ValueError: If the specified top_k is not > 0.
        """
        if not isinstance(document_store, MemoryDocumentStore):
            raise ValueError("document_store must be an instance of MemoryDocumentStore")

        self.document_store = document_store

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Run the MemoryRetriever on the given input data.

        :param query: The query string for the retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the BM25 scores or not.
        :param document_stores: A dictionary mapping DocumentStore names to instances.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not a MemoryDocumentStore instance.
        """
        if filters is None:
            filters = self.filters
        if top_k is None:
            top_k = self.top_k
        if scale_score is None:
            scale_score = self.scale_score

        docs = []
        for query in queries:
            docs.append(
                self.document_store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score)
            )
        return {"documents": docs}
