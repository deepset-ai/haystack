# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore

from haystack import Document, component
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever as InMemoryEmbeddingRetrieverBase
from haystack.document_stores.types import FilterPolicy


@component
class InMemoryEmbeddingRetriever(InMemoryEmbeddingRetrieverBase):
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
    from haystack_experimental.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack_experimental.document_stores.in_memory import InMemoryDocumentStore

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

        super(InMemoryEmbeddingRetriever, self).__init__(
            document_store=document_store,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
            filter_policy=filter_policy,
        )

    @component.output_types(documents=List[Document])
    async def run_async(  # pylint: disable=too-many-positional-arguments
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

        docs = await self.document_store.embedding_retrieval_async(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
        )

        return {"documents": docs}
