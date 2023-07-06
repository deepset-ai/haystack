from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from haystack.preview import component, Document, ComponentInput, ComponentOutput
from haystack.preview.document_stores import MemoryDocumentStore


@component
class MemoryRetriever:
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.
    """

    @dataclass
    class Input(ComponentInput):
        """
        Input data for the MemoryRetriever component.

        :param query: The query string for the retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the BM25 scores or not.
        :param stores: A dictionary mapping document store names to instances.
        """

        query: str
        filters: Dict[str, Any]
        top_k: int
        scale_score: bool
        stores: Dict[str, Any]

    @dataclass
    class Output(ComponentOutput):
        """
        Output data from the MemoryRetriever component.

        :param documents: The retrieved documents.
        """

        documents: List[Document]

    def __init__(
        self,
        document_store_name: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
    ):
        """
        Create a MemoryRetriever component.

        :param document_store_name: The name of the MemoryDocumentStore to retrieve documents from.
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :param scale_score: Whether to scale the BM25 score or not (default is True).

        :raises ValueError: If the specified top_k is not > 0.
        """
        self.document_store_name = document_store_name
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.defaults = {"top_k": top_k, "scale_score": scale_score, "filters": filters or {}}

    def run(self, data: Input) -> Output:
        """
        Run the MemoryRetriever on the given input data.

        :param data: The input data for the retriever.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        if self.document_store_name not in data.stores:
            raise ValueError(
                f"MemoryRetriever's document store '{self.document_store_name}' not found "
                f"in input stores {list(data.stores.keys())}"
            )
        document_store = data.stores[self.document_store_name]
        if not isinstance(document_store, MemoryDocumentStore):
            raise ValueError("MemoryRetriever can only be used with a MemoryDocumentStore instance.")
        docs = document_store.bm25_retrieval(
            query=data.query, filters=data.filters, top_k=data.top_k, scale_score=data.scale_score
        )
        return MemoryRetriever.Output(documents=docs)
