from typing import Dict, List, Any, Optional

from haystack.preview import component, Document, default_to_dict, default_from_dict, DeserializationError
from haystack.preview.document_stores import MemoryDocumentStore, document_store


@component
class MemoryEmbeddingRetriever:
    """
    A component for retrieving documents from a MemoryDocumentStore using a vector similarity metric.

    Needs to be connected to a MemoryDocumentStore to run.
    """

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ):
        """
        Create a MemoryEmbeddingRetriever component.

        :param document_store: An instance of MemoryDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Default is None.
        :param top_k: The maximum number of documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents or not. Default is True.
        :param return_embedding: Whether to return the embedding of the retrieved Documents. Default is False.

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
        self.return_embedding = return_embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            filters=self.filters,
            top_k=self.top_k,
            scale_score=self.scale_score,
            return_embedding=self.return_embedding,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEmbeddingRetriever":
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
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the MemoryEmbeddingRetriever on the given input data.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the scores of the retrieved documents or not.
        :param return_embedding: Whether to return the embedding of the retrieved Documents.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not a MemoryDocumentStore instance.
        """
        if filters is None:
            filters = self.filters
        if top_k is None:
            top_k = self.top_k
        if scale_score is None:
            scale_score = self.scale_score
        if return_embedding is None:
            return_embedding = self.return_embedding

        docs = self.document_store.embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
        )

        return {"documents": docs}
