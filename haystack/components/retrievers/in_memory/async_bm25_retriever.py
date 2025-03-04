# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore

from haystack import Document, component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever as InMemoryBM25RetrieverBase
from haystack.document_stores.types import FilterPolicy


@component
class InMemoryBM25Retriever(InMemoryBM25RetrieverBase):
    """
    Retrieves documents that are most similar to the query using keyword-based algorithm.

    Use this retriever with the InMemoryDocumentStore.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_experimental.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore

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

    def __init__(  # pylint: disable=too-many-positional-arguments
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

        super(InMemoryBM25Retriever, self).__init__(
            document_store=document_store,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            filter_policy=filter_policy,
        )

    @component.output_types(documents=List[Document])
    async def run_async(
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

        docs = await self.document_store.bm25_retrieval_async(
            query=query, filters=filters, top_k=top_k, scale_score=scale_score
        )
        return {"documents": docs}
