from typing import Dict, List, Any, Optional

from haystack.preview import component, Document, default_to_dict, default_from_dict, DeserializationError
from haystack.preview.document_stores import InMemoryDocumentStore, document_store


@component
class InMemoryBM25Retriever:
    """
    Uses the BM25 algorithm to retrieve documents from the InMemoryDocumentStore.

    Needs to be connected to the InMemoryDocumentStore to run.
    """

    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
    ):
        """
        Create the InMemoryBM25Retriever component.

        :param document_store: An instance of InMemoryDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Defaults to `None`.
        :param top_k: The maximum number of documents to retrieve. Defaults to `10`.
        :param scale_score: Scales the BM25 score to a unit interval in the range of 0 to 1, where 1 means extremely relevant. If set to `False`, uses raw similarity scores.
        Defaults to `False`.

        :raises ValueError: If the specified `top_k` is not > 0.
        """
        if not isinstance(document_store, InMemoryDocumentStore):
            raise ValueError("document_store must be an instance of InMemoryDocumentStore")

        self.document_store = document_store

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, the top_k is {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"document_store": type(self.document_store).__name__}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(
            self, document_store=docstore, filters=self.filters, top_k=self.top_k, scale_score=self.scale_score
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryBM25Retriever":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")
        if init_params["document_store"]["type"] not in document_store.registry:
            raise DeserializationError(f"DocumentStore type '{init_params['document_store']['type']}' not found")

        docstore_class = document_store.registry[init_params["document_store"]["type"]]
        docstore = docstore_class.from_dict(init_params["document_store"])
        data["init_parameters"]["document_store"] = docstore
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Run the InMemoryBM25Retriever on the given input data.

        :param query: The query string for the Retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Scales the BM25 score to a unit interval in the range of 0 to 1, where 1 means extremely relevant. If set to `False`, uses raw similarity scores.
            If not specified, the value provided at initialization is used.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.
        """
        if filters is None:
            filters = self.filters
        if top_k is None:
            top_k = self.top_k
        if scale_score is None:
            scale_score = self.scale_score

        docs = self.document_store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score)
        return {"documents": docs}
