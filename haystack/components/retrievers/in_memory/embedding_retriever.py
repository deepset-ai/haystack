# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import FilterPolicy


@component
class InMemoryEmbeddingRetriever:
    """
    Retrieves documents that are most semantically similar to the query.

    Use this retriever with the InMemoryDocumentStore.

    When using this retriever, make sure it has query and document embeddings available.
    In indexing pipelines, use a DocumentEmbedder to embed documents.
    In query pipelines, use a TextEmbedder to embed queries and send them to the retriever.

    ### Usage example
    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    docs = [
        Document(content="Python is a popular programming language"),
        Document(content="python ist eine beliebte Programmiersprache"),
    ]
    doc_embedder = SentenceTransformersDocumentEmbedder()
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)["documents"]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs_with_embeddings)
    retriever = InMemoryEmbeddingRetriever(doc_store)

    query="Programmiersprache"
    text_embedder = SentenceTransformersTextEmbedder()
    text_embedder.warm_up()
    query_embedding = text_embedder.run(query)["embedding"]

    result = retriever.run(query_embedding=query_embedding)

    print(result["documents"])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        document_store: InMemoryDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        Create the InMemoryEmbeddingRetriever component.

        :param document_store:
            An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
        :param filters:
            A dictionary with filters to narrow down the retriever's search space in the document store.
        :param top_k:
            The maximum number of documents to retrieve.
        :param scale_score:
            When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
            When `False`, uses raw similarity scores.
        :param return_embedding:
            When `True`, returns the embedding of the retrieved documents.
            When `False`, returns just the documents, without their embeddings.
        :param filter_policy: The filter policy to apply during retrieval.
        Filter policy determines how filters are applied when retrieving documents. You can choose:
        - `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
        Use this policy to dynamically change filtering for specific queries.
        - `MERGE`: Combines runtime filters with initialization filters to narrow down the search.
        :raises ValueError:
            If the specified top_k is not > 0.
        """
        if not isinstance(document_store, InMemoryDocumentStore):
            raise ValueError("document_store must be an instance of InMemoryDocumentStore")

        self.document_store = document_store

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, top_k is {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score
        self.return_embedding = return_embedding
        self.filter_policy = filter_policy

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"document_store": type(self.document_store).__name__}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            filters=self.filters,
            top_k=self.top_k,
            scale_score=self.scale_score,
            return_embedding=self.return_embedding,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")
        if "filter_policy" in init_params:
            init_params["filter_policy"] = FilterPolicy.from_str(init_params["filter_policy"])
        data["init_parameters"]["document_store"] = InMemoryDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the InMemoryEmbeddingRetriever on the given input data.

        :param query_embedding:
            Embedding of the query.
        :param filters:
            A dictionary with filters to narrow down the search space when retrieving documents.
        :param top_k:
            The maximum number of documents to return.
        :param scale_score:
            When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
            When `False`, uses raw similarity scores.
        :param return_embedding:
            When `True`, returns the embedding of the retrieved documents.
            When `False`, returns just the documents, without their embeddings.
        :returns:
            The retrieved documents.

        :raises ValueError:
            If the specified DocumentStore is not found or is not an InMemoryDocumentStore instance.
        """
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters
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
