# pylint: disable=too-many-instance-attributes

import math

from typing import Any, List, Optional, Dict, Union, Tuple
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from haystack.nodes import BaseComponent, BaseRetriever, EmbeddingRetriever
from haystack.document_stores import KeywordDocumentStore
from haystack.schema import Document, MultiLabel


logger = logging.getLogger(__name__)


class FileSimilarityRetriever(BaseComponent):
    """
    This retriever performs retrieval for file similarity. It is a self-referential retriever that will use existing
    files as a query and returns a list of the most similar documents from each file in the order of similarity to the
    query file. It uses reciprocal rank fusion to determine file similarity. That means, it uses each document from the
    query file and performs a retrieval for this document. It then aggregates the results from each document query. A
    similar approach is described here: https://arxiv.org/pdf/2201.01614.pdf (Althammer et al. 2022).
    """

    outgoing_edges = 1

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        document_store: KeywordDocumentStore,
        primary_retriever: Optional[BaseRetriever] = None,
        secondary_retriever: Optional[BaseRetriever] = None,
        file_aggregation_key: str = "file_id",
        keep_original_score: Optional[str] = None,
        top_k: int = 10,
        max_query_len: int = 6000,
        max_num_queries: Optional[int] = None,
        use_existing_embedding: bool = True,
    ) -> None:
        """
        Initialize an instance of FileSimilarityRetriever.
        :param document_store: The document store that the retriever should retrieve from.
        :param file_aggregation_key: The meta data key that should be used to aggregate documents to the file level.
        :param primary_retriever: A clutch until haystack supports passing a list of retrievers.
        :param secondary_retriever: A clutch until haystack supports passing a list of retrievers.
        :param keep_original_score: Set this to store the original score of the returned document in the document's meta
            field. The document's score property will be replaced with the reciprocal rank fusion score.
        :param top_k: How many documents to return.
        :param max_query_len: How many chars can be in a query document. The document will be cut off if it is longer
            than the maximum length. We need this here because there might be an issue with queries that are too long
            and the BM25Retriever because an error will be thrown if the query excees the `max_clause_count` search
            setting (https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-settings.html)
        :param max_num_queries: The maximum number of queries that should be run for a single file. If the number of
            query documents exceeds this limit, the query documents will be split into n parts so that
            n < max_num_queries and every nth document will be kept.
        :param use_existing_embeddings: Whether to re-use the existing embeddings from the index.
            To optimize speed for the file similarity retrieval you should set this parameter to `True`.
            This way the FileSimilarityRetriever can run on the CPU.
        """
        super().__init__()
        self.retrievers = []
        if primary_retriever:
            self.retrievers.append(primary_retriever)

        if secondary_retriever:
            self.retrievers.append(secondary_retriever)

        self.file_aggregation_key = file_aggregation_key

        self.document_store = document_store
        self.keep_original_score = keep_original_score
        self.top_k = top_k
        self.max_query_len = max_query_len
        self.max_num_queries = max_num_queries
        self.use_existing_embedding = use_existing_embedding
        self.executor = ThreadPoolExecutor(max_workers=len(self.retrievers))

    def run(
        self,
        query: Union[str, List[str], None] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[Dict[Any, Any]] = None,
        top_k: Optional[int] = None,  # Additional parameter with a default value
        indices: Optional[Union[str, List[Union[str, None]]]] = None,  # Additional parameter with a default value
        filters: Optional[Dict[Any, Any]] = None,  # Additional parameter with a default value
        file_index: Optional[str] = None  # Additional parameter with a default value
    ) -> Tuple[Dict[Any, Any], str]:
        """
        Performs file similarity retrieval using all retrievers that this node was initialized with.
        The query should be the file aggregator value that will be used to get all relevant documents from the
        document_store.

        :param query: Will be used to filter for all documents that belong to the file you want to use as query file.
        :param top_k: The maximum number of documents to return.
        :param indices: The document_store index or indices that the retrievers should retrieve from.
        :param file_index: The index that the query file should be retrieved from.
        :param filters: Filters that should be applied for each retriever.
        """
        if isinstance(query, list):
            # Handle the case where query is a list
            # For example, take the first element or concatenate
            query = query[0]  # or your own logic to handle a list

        if isinstance(indices, list):
            # Convert indices to the simpler type if necessary
            indices = [index for index in indices if index is not None]  # Remove None values

        if query is None:
            raise ValueError("Query cannot be None.")
        
        if query is not None:
            retrieved_docs = self.retrieve(
                query=query, top_k=top_k, indices=indices, file_index=file_index, filters=filters
            )

        return {"documents": retrieved_docs}, "output_1"

    def run_batch(
        self,
        queries: Union[str, List[str], None] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[Dict[Any, Any]] = None,
        debug: Optional[bool] = None,
        # Additional parameters with default values
        top_k: Optional[int] = None,
        indices: Optional[Union[str, List[Union[str, None]]]] = None,
        filters: Optional[Dict[Any, Any]] = None,
        file_index: Optional[str] = None
    ) -> Any:        
        # Convert complex types to simpler types
        if queries is not None:
            simple_queries = [q[0] if isinstance(q, list) else q for q in queries]  # Assuming you want the first query if it's a list of lists
        
        results = []
        for query in simple_queries:
            results.append(
                self.retrieve(query=query, top_k=top_k, indices=indices, filters=filters, file_index=file_index)
            )

        return {"documents": results}, "output_1"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        indices: Optional[Union[str, List[Optional[str]]]] = None,
        file_index: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Performs file similarity retrieval using all retrievers that this node was initialized with.
        The query should be the file aggregator value that will be used to get all relevant documents from the
        document_store.

        :param query: Will be used to filter for all documents that belong to the file you want to use as query file.
        :param top_k: The maximum number of documents to return.
        :param indices: The document_store index or indices that the retrievers should retrieve from.
        :param file_index: The index that the query file should be retrieved from.
        :param filters: Filters that should be applied for each retriever.
        """
        if isinstance(indices, str) or indices is None:
            retriever_indices = [indices] * len(self.retrievers)
        else:
            retriever_indices = indices

        if not top_k:
            top_k = self.top_k

        query_file_documents = self._get_documents(file_filter=query, index=file_index)

        if self.max_num_queries is not None and len(query_file_documents) > self.max_num_queries:
            logger.warning(
                "Query file %s has %s documents. "
                "It exceeds the maximum number of query documents and was reduced to %s query documents.",
                query,
                len(query_file_documents),
                self.max_num_queries,
            )
            num_splits = math.ceil(len(query_file_documents) / self.max_num_queries)
            query_file_documents = query_file_documents[::num_splits]

        if len(query_file_documents):
            retrieved_docs = []
            futures = []
            for idx, retriever in zip(retriever_indices, self.retrievers):
                if isinstance(retriever, EmbeddingRetriever) and all(
                    doc.embedding is not None for doc in query_file_documents
                ):
                    future = self.executor.submit(
                        self._retrieve_for_documents_by_embedding,
                        retriever=retriever,
                        documents=query_file_documents,
                        index=idx,
                        filters=filters,
                    )
                else:
                    future = self.executor.submit(
                        self._retrieve_for_documents,
                        retriever=retriever,
                        documents=query_file_documents,
                        index=idx,
                        filters=filters,
                    )
                futures.append(future)

            for future in as_completed(futures):
                retrieved_docs.extend(future.result())

            aggregated_results = self._aggregate_results(results=retrieved_docs, top_k=top_k)
        else:
            logger.info("Could not find any indexed documents for query: %s.", query)
            aggregated_results = []

        return aggregated_results

    def _get_documents(self, file_filter: str, index: Optional[str]) -> List[Document]:
        docs: List[Document] = self.document_store.get_all_documents(
            index=index,
            filters={self.file_aggregation_key: [file_filter]},
            return_embedding=self.use_existing_embedding,
        )
        return docs

    def _retrieve_for_documents_by_embedding(
        self,
        retriever: EmbeddingRetriever,
        documents: List[Document],
        index: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[List[Document]]:
        doc_store = retriever.document_store
        if doc_store is None:
            raise ValueError("Document store cannot be None")

        top_k = retriever.top_k

        # Filter out documents where the embedding is None
        valid_documents = [doc for doc in documents if doc.embedding is not None]
        if not valid_documents:
            raise ValueError("No valid document embeddings found for query.")

        # Filter out documents where the embedding is None and create the query_embs list
        # redundant check to pass mypy linter
        query_embs = [doc.embedding for doc in valid_documents if doc.embedding is not None]
        if not query_embs:
            raise ValueError("All document embeddings are None")

        results: List[List[Document]] = doc_store.query_by_embedding_batch(
            query_embs=query_embs, filters=filters, index=index, top_k=top_k
        )
        return results

    def _retrieve_for_documents(
        self,
        retriever: BaseRetriever,
        documents: List[Document],
        index: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[List[Document]]:
        queries = []
        for doc in documents:
            content = doc.content[: self.max_query_len]
            queries.append(content)
            if len(content) != len(doc.content):
                logger.warning(
                    "Document %s retrieved with aggregation key '%s' exceeds max_query_len of %s and was cut off.",
                    doc,
                    self.file_aggregation_key,
                    self.max_query_len,
                )

        docs: List[List[Document]] = retriever.retrieve_batch(
            queries=queries, index=index, filters=filters, scale_score=False
        )

        return docs

    def _aggregate_results(self, results: List[List[Document]], top_k: int) -> List[Document]:
        # We iterate over each result list that contains for each query document and each retriever a list of documents
        # ranked by their similarity to the query document.
        # We group the result documents by the same file aggregation key and calculate the reciprocal rank fusion score.
        aggregator_doc_lookup = defaultdict(list)
        aggregated_scores: Dict = defaultdict(int)
        for result in results:
            for idx, doc in enumerate(result):
                aggregator = doc.meta.get(self.file_aggregation_key)
                if aggregator is None:
                    logger.warning(
                        "Document %s can not be aggregated. Missing aggregation key '%s' in meta.",
                        doc,
                        self.file_aggregation_key,
                    )
                else:
                    score = self._calculate_reciprocal_rank_fusion(idx)
                    aggregated_scores[doc.meta[self.file_aggregation_key]] += score
                    aggregator_doc_lookup[doc.meta[self.file_aggregation_key]].append(doc)

        # For each aggregated file we want to sort the retrieved documents by their score so that we
        # can return the most relevant document for each aggregation later.
        for aggregator in aggregator_doc_lookup:
            aggregator_doc_lookup[aggregator] = sorted(
                aggregator_doc_lookup[aggregator], key=lambda doc: doc.score, reverse=True  # type: ignore
            )

        sorted_aggregator_scores = sorted(aggregated_scores.items(), key=lambda d: d[1], reverse=True)  # type: ignore
        result_docs = []
        for aggregator_id, rrf_score in sorted_aggregator_scores[:top_k]:
            doc = aggregator_doc_lookup[aggregator_id][0]
            if self.keep_original_score:
                doc.meta[self.keep_original_score] = doc.score
            doc.score = rrf_score
            result_docs.append(doc)

        return result_docs

    @staticmethod
    def _calculate_reciprocal_rank_fusion(current_idx: int) -> float:
        """
        Calculates the reciprocal rank score for a Document instance at the current rank.
        See https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        :param current_idx: The rank position of a document in the result list.
        """
        # The paper above mentions a constant of 60 that should be used in the denominator.
        # The denominator is the result of adding this constant and the rank of the retrieved document
        # We set the constant to 61 because python passes the rank starting from 0.
        reciprocal_rank_constant = 61
        return 1 / (reciprocal_rank_constant + current_idx)
