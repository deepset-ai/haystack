from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from haystack.preview import component, Document, ComponentInput, ComponentOutput
from haystack.preview.document_stores import MemoryDocumentStore, StoreComponent


@component
class MemoryRetriever(StoreComponent):
    """
    A component for retrieving documents from a MemoryDocumentStore using the BM25 algorithm.

    Needs to be connected to a MemoryDocumentStore to run.
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

    @dataclass
    class Output(ComponentOutput):
        """
        Output data from the MemoryRetriever component.

        :param documents: The retrieved documents.
        """

        documents: List[Document]

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

    @property
    def store(self) -> Optional[MemoryDocumentStore]:
        """
        This property allows Pipelines to connect the component with the stores it requires.

        Stores have to be instances of MemoryDocumentStore, or the assignment will fail.
        """
        return getattr(self, "_store", None)

    @store.setter
    def store(self, store: MemoryDocumentStore):
        """
        This property allows Pipelines to connect the component with the stores it requires.

        Stores have to be instances of MemoryDocumentStore, or the assignment will fail.

        :param store: the MemoryDocumentStore instance to retrieve from.
        :raises ValueError if the store is not an instance of MemoryDocumentStore.
        """
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        if not isinstance(store, MemoryDocumentStore):
            raise ValueError("MemoryRetriever can only be used with a MemoryDocumentStore instance.")
        self._store = store

    def warmup(self):
        """
        Double-checks that a store is set before running this component in a pipeline.
        """
        if not self.store:
            raise ValueError(
                "MemoryRetriever needs a store to run: " "use the 'store' parameter of 'add_component' to connect them."
            )

    def run(self, data: Input) -> Output:
        """
        Run the MemoryRetriever on the given input data.

        :param data: The input data for the retriever.
        :return: The retrieved documents.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        if not self.store:
            raise ValueError("MemoryRetriever needs a store to run: set the store instance to the self.store attribute")
        docs = self.store.bm25_retrieval(
            query=data.query, filters=data.filters, top_k=data.top_k, scale_score=data.scale_score
        )
        return MemoryRetriever.Output(documents=docs)
