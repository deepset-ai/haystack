from typing import Dict, List, Any, Optional

from haystack import component, Document, default_to_dict, default_from_dict, DeserializationError
from haystack.document_stores.types import DocumentStore


@component
class FilterRetriever:
    """
    Retrieves documents that match the provided filters.
    """

    def __init__(
        self, document_store: DocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ):
        """
        Create the FilterRetriever component.

        :param document_store: An instance of InMemoryDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Defaults to `None`.
        :param top_k: The maximum number of documents to retrieve. Defaults to `None` in which case all documents are retrieved.

        :raises ValueError: If the `top_k` is specified but is not > 0.
        """
        self.document_store = document_store

        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, the top_k is {top_k}")

        self.filters = filters
        self.top_k = top_k

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
        return default_to_dict(self, document_store=docstore, filters=self.filters, top_k=self.top_k)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterRetriever":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")
        data["init_parameters"]["document_store"] = FilterRetriever.from_dict(data["init_parameters"]["document_store"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """
        Run the FilterRetriever on the given input data.

        :param query: The query string for the Retriever. It is ignored by the retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found
        """
        if filters is None:
            filters = self.filters
        if top_k is None:
            top_k = self.top_k

        docs = self.document_store.filter_documents(filters=filters)
        if top_k is not None:
            docs = docs[:top_k]
        return {"documents": docs}
