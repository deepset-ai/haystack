# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import FilterPolicy


@component
class InMemoryBM25Retriever:
    """
    Retrieves documents that are most similar to the query using keyword-based algorithm.

    Use this retriever with the InMemoryDocumentStore.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    docs = [
        Document(content="Python is a popular programming language"),
        Document(content="python ist eine beliebte Programmiersprache"),
    ]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    retriever = InMemoryBM25Retriever(doc_store)

    result = retriever.run(query="Programmiersprache")

    print(result["documents"])
    ```
    """

    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        Create the InMemoryBM25Retriever component.

        :param document_store:
            An instance of InMemoryDocumentStore where the retriever should search for relevant documents.
        :param filters:
            A dictionary with filters to narrow down the retriever's search space in the document store.
        :param top_k:
            The maximum number of documents to retrieve.
        :param scale_score:
            When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
            When `False`, uses raw similarity scores.
        :param filter_policy: The filter policy to apply during retrieval.
        Filter policy determines how filters are applied when retrieving documents. You can choose:
        - `REPLACE` (default): Overrides the initialization filters with the filters specified at runtime.
        Use this policy to dynamically change filtering for specific queries.
        - `MERGE`: Combines runtime filters with initialization filters to narrow down the search.
        :raises ValueError:
            If the specified `top_k` is not > 0.
        """
        if not isinstance(document_store, InMemoryDocumentStore):
            raise ValueError("document_store must be an instance of InMemoryDocumentStore")

        self.document_store = document_store

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, the top_k is {top_k}")

        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score
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
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryBM25Retriever":
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
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Run the InMemoryBM25Retriever on the given input data.

        :param query:
            The query string for the Retriever.
        :param filters:
            A dictionary with filters to narrow down the search space when retrieving documents.
        :param top_k:
            The maximum number of documents to return.
        :param scale_score:
            When `True`, scales the score of retrieved documents to a range of 0 to 1, where 1 means extremely relevant.
            When `False`, uses raw similarity scores.
        :returns:
            The retrieved documents.

        :raises ValueError:
            If the specified DocumentStore is not found or is not a InMemoryDocumentStore instance.
        """
        if self.filter_policy == FilterPolicy.MERGE and filters:
            filters = {**(self.filters or {}), **filters}
        else:
            filters = filters or self.filters
        if top_k is None:
            top_k = self.top_k
        if scale_score is None:
            scale_score = self.scale_score

        docs = self.document_store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score)
        return {"documents": docs}
