# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional

from haystack.dataclasses import Document
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore as InMemoryDocumentStoreBase
from haystack.document_stores.types import DuplicatePolicy


class InMemoryDocumentStore(InMemoryDocumentStoreBase):
    """
    Asynchronous version of the in-memory document store.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25L",
        bm25_parameters: Optional[Dict] = None,
        embedding_similarity_function: Literal["dot_product", "cosine"] = "dot_product",
        index: Optional[str] = None,
        async_executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initializes the DocumentStore.

        :param bm25_tokenization_regex: The regular expression used to tokenize the text for BM25 retrieval.
        :param bm25_algorithm: The BM25 algorithm to use. One of "BM25Okapi", "BM25L", or "BM25Plus".
        :param bm25_parameters: Parameters for BM25 implementation in a dictionary format.
            For example: {'k1':1.5, 'b':0.75, 'epsilon':0.25}
            You can learn more about these parameters by visiting https://github.com/dorianbrown/rank_bm25.
        :param embedding_similarity_function: The similarity function used to compare Documents embeddings.
            One of "dot_product" (default) or "cosine". To choose the most appropriate function, look for information
            about your embedding model.
        :param index: A specific index to store the documents. If not specified, a random UUID is used.
            Using the same index allows you to store documents across multiple InMemoryDocumentStore instances.
        :param async_executor:
            Optional ThreadPoolExecutor to use for async calls. If not provided, a single-threaded
            executor will initialized and used.
        """
        super().__init__(
            bm25_tokenization_regex=bm25_tokenization_regex,
            bm25_algorithm=bm25_algorithm,
            bm25_parameters=bm25_parameters,
            embedding_similarity_function=embedding_similarity_function,
            index=index,
        )

        self.executor = (
            ThreadPoolExecutor(thread_name_prefix=f"async-inmemory-docstore-executor-{id(self)}", max_workers=1)
            if async_executor is None
            else async_executor
        )

    async def count_documents_async(self) -> int:
        """
        Returns the number of how many documents are present in the DocumentStore.
        """
        return len(self.storage.keys())

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol
        documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.filter_documents(filters=filters)
        )

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Refer to the DocumentStore.write_documents() protocol documentation.

        If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.write_documents(documents=documents, policy=policy)
        )

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        await asyncio.get_event_loop().run_in_executor(
            self.executor, lambda: self.delete_documents(document_ids=document_ids)
        )

    async def bm25_retrieval_async(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, scale_score: bool = False
    ) -> List[Document]:
        """
        Retrieves documents that are most relevant to the query using BM25 algorithm.

        :param query: The query string.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved documents. Default is False.
        :returns: A list of the top_k documents most relevant to the query.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score),
        )

    async def embedding_retrieval_async(  # pylint: disable=too-many-positional-arguments
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved Documents. Default is False.
        :param return_embedding: Whether to return the embedding of the retrieved Documents. Default is False.
        :returns: A list of the top_k documents most relevant to the query.
        """
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                scale_score=scale_score,
                return_embedding=return_embedding,
            ),
        )
